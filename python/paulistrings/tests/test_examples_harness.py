"""Tests for `examples/common/harness.py` (handoff item P0d).

`examples/` is not a package under `python/`, so — following
`test_examples_report.py` — this file puts the repo's `examples/` directory on
`sys.path` and imports `harness` as a member of the `common` package. Nothing
here needs matplotlib, stim or qiskit: the harness is numpy-only (numpy arrives
with `paulistrings` itself), so the whole file runs in CI.

The tiny scenario every timing/sweep test uses is exact by hand. With
`obs = Z₀` and a one-gate circuit `rx(θ, 0)`, the Heisenberg-evolved operator is

    cos θ · Z₀ + sin θ · Y₀        (θ = 0.5: 0.87758…, 0.47942…)

and `⟨Y⟩ = 0` in `|0…0⟩`, so the expectation is `cos θ` **whether or not the
`Y₀` term survives truncation**. That gives three hand-predictable regimes for
a `min_abs_coeff` sweep: `eps = 0.9` drops both terms (inclusive boundary) and
the expectation collapses to 0; `eps = 0.5` drops only `sin θ` and the answer
is exact; `eps = 0.1` keeps both and the answer is exact.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import harness, report  # noqa: E402

THETA = 0.5
COS_THETA = math.cos(THETA)  # 0.8775825618903728
SIN_THETA = math.sin(THETA)  # 0.479425538604203
NUM_QUBITS = 2


def _observable() -> PauliSum:
    return PauliSum.from_strings({"ZI": 1.0}, num_qubits=NUM_QUBITS)


def _circuit() -> Circuit:
    c = Circuit(NUM_QUBITS)
    c.rx(THETA, 0)
    return c


def _evolved(policy=None) -> PauliSum:
    return _observable().propagate(_circuit(), policy, direction="heisenberg")


# --- make_policy (A7) ----------------------------------------------------


def test_make_policy_none_when_no_knobs():
    assert harness.make_policy() is None
    assert harness.make_policy(None, None) is None


def test_make_policy_weight_only():
    policy = harness.make_policy(max_weight=1)
    assert repr(policy) == "Truncation(Weight(1))"
    # Weight 1 keeps both single-qubit terms of the evolved operator.
    assert len(_evolved(policy)) == 2
    # Weight 0 keeps only the identity, of which the evolved operator has none.
    assert len(_evolved(harness.make_policy(max_weight=0))) == 0


def test_make_policy_coeff_only():
    policy = harness.make_policy(min_abs_coeff=0.5)
    assert repr(policy) == "Truncation(Coeff(0.5))"
    # The inclusive boundary this docstring documents: |c| <= eps is dropped,
    # so sin(0.5) = 0.4794... goes and cos(0.5) = 0.8775... stays.
    survivors = _evolved(policy).coefficients()
    assert len(survivors) == 1
    assert survivors[0].real == pytest.approx(COS_THETA, abs=1e-15)


def test_make_policy_and_composition():
    policy = harness.make_policy(max_weight=3, min_abs_coeff=1e-6)
    assert repr(policy) == "Truncation(And(Weight(3), Coeff(1e-6)))"


def test_make_policy_composition_is_the_conjunction():
    # Weight cap alone keeps 2 terms, coeff cap alone keeps 1; the conjunction
    # must keep the intersection, i.e. 1.
    assert len(_evolved(harness.make_policy(max_weight=1))) == 2
    assert len(_evolved(harness.make_policy(min_abs_coeff=0.5))) == 1
    assert len(_evolved(harness.make_policy(max_weight=1, min_abs_coeff=0.5))) == 1
    assert len(_evolved(harness.make_policy(max_weight=0, min_abs_coeff=0.5))) == 0


def test_make_policy_boundary_is_inclusive():
    # A coefficient exactly equal to the cutoff is dropped, not kept.
    assert len(_evolved(harness.make_policy(min_abs_coeff=SIN_THETA))) == 1
    assert len(_evolved(harness.make_policy(min_abs_coeff=math.nextafter(SIN_THETA, 0.0)))) == 2


def test_make_policy_rejects_bad_knobs():
    with pytest.raises(ValueError, match="max_weight"):
        harness.make_policy(max_weight=-1)
    with pytest.raises(ValueError, match="max_weight"):
        harness.make_policy(max_weight=2.5)
    with pytest.raises(ValueError, match="min_abs_coeff"):
        harness.make_policy(min_abs_coeff=-1e-6)


def test_make_policy_offers_no_topn():
    # Banned from comparative runs (plan D3); it must not be reachable here.
    assert not hasattr(harness, "make_topn_policy")
    with pytest.raises(TypeError):
        harness.make_policy(topn=4)


# --- TruncationSpec ------------------------------------------------------


def test_truncation_spec_coercion_forms():
    assert harness.TruncationSpec.coerce(None) == harness.TruncationSpec()
    assert harness.TruncationSpec.coerce((4, 1e-6)) == harness.TruncationSpec(4, 1e-6)
    assert harness.TruncationSpec.coerce({"min_abs_coeff": 1e-6}) == harness.TruncationSpec(
        None, 1e-6
    )
    spec = harness.TruncationSpec(2, 1e-3)
    assert harness.TruncationSpec.coerce(spec) is spec


def test_truncation_spec_as_dict_omits_unset_knobs():
    assert harness.TruncationSpec().as_dict() == {}
    assert harness.TruncationSpec(min_abs_coeff=1e-6).as_dict() == {"min_abs_coeff": 1e-6}
    assert harness.TruncationSpec(4, 1e-6).as_dict() == {
        "max_weight": 4,
        "min_abs_coeff": 1e-6,
    }


def test_truncation_spec_rejects_unknown_and_opaque_forms():
    with pytest.raises(ValueError, match="topn"):
        harness.TruncationSpec.coerce({"topn": 10})
    with pytest.raises(ValueError, match="max_weight, min_abs_coeff"):
        harness.TruncationSpec.coerce((1, 2, 3))
    with pytest.raises(TypeError, match="truncation knobs"):
        harness.TruncationSpec.coerce(truncation.coeff(1e-6))


# --- Thread discipline (A7) ----------------------------------------------


def _is_pinned() -> bool:
    import os

    gained = harness.rayon_worker_estimate()
    return os.environ.get("RAYON_NUM_THREADS") == "1" and (
        gained is None or gained <= harness.PINNED_RAYON_THREADS
    )


def test_assert_single_threaded_agrees_with_the_environment():
    """Whichever way this process runs, the assert must agree with reality.

    The Rayon pool is process-global and built once, so a test cannot create
    the other case by setting `os.environ` — hence the branch rather than a
    parametrization. Run the suite under `RAYON_NUM_THREADS=1` to exercise the
    passing side; a plain `pytest` run exercises the raising side.
    """
    if _is_pinned():
        harness.assert_single_threaded()
    else:
        with pytest.raises(RuntimeError, match="RAYON_NUM_THREADS"):
            harness.assert_single_threaded()


def test_thread_pin_error_names_the_fix():
    import os

    if _is_pinned():
        pytest.skip("this process is pinned; the failure message is not reachable")
    with pytest.raises(RuntimeError) as excinfo:
        harness.assert_single_threaded()
    message = str(excinfo.value)
    assert "RAYON_NUM_THREADS=1 python" in message
    assert "before the interpreter starts" in message
    assert os.environ.get("RAYON_NUM_THREADS", "unset") in message or "None" in message


def test_import_thread_baseline_is_readable_on_linux():
    # /proc is present on every host this suite targets; the helpers must
    # nonetheless degrade to None rather than raise elsewhere.
    assert harness.IMPORT_THREAD_COUNT is None or harness.IMPORT_THREAD_COUNT >= 1
    gained = harness.rayon_worker_estimate()
    assert gained is None or gained >= 0


def test_memory_probes_read_procfs():
    peak = harness.peak_memory_kb()
    current = harness.current_memory_kb()
    assert peak is None or peak > 0
    assert current is None or current > 0
    if peak is not None and current is not None:
        assert peak >= current


# --- Logging discipline --------------------------------------------------


def test_logging_guard_blocks_a_debug_enabled_timed_run():
    logger = logging.getLogger("paulistrings.propagate")
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    paulistrings.reset_log_cache()
    try:
        assert not harness.logging_is_quiet()
        with pytest.raises(RuntimeError, match="DEBUG-enabled"):
            harness.assert_logging_quiet()
        with pytest.raises(RuntimeError, match="DEBUG-enabled"):
            harness.run_propagation(
                _circuit(), _observable(), None, "heisenberg", state="z+"
            )
        # Explicit opt-out still runs (diagnostic-only timings).
        record = harness.run_propagation(
            _circuit(),
            _observable(),
            None,
            "heisenberg",
            state="z+",
            require_quiet_logging=False,
        )
        assert record.final_terms == 2
    finally:
        logger.setLevel(previous)
        paulistrings.reset_log_cache()


def test_logging_is_quiet_by_default():
    assert harness.logging_is_quiet()


# --- run_propagation -----------------------------------------------------


def test_run_propagation_record_is_complete():
    record = harness.run_propagation(
        _circuit(),
        _observable(),
        (None, 1e-9),
        "heisenberg",
        state="z+",
        oracle_value=COS_THETA,
        seeds={"circuit": 7},
        extra={"theta": THETA},
    )

    assert isinstance(record, report.RunRecord)
    assert record.engine == "paulistrings"
    assert record.engine_version != ""
    assert record.n_qubits == NUM_QUBITS
    assert record.direction == "heisenberg"
    assert record.truncation == {"min_abs_coeff": 1e-9}
    assert record.propagation_time_s > 0.0
    assert record.contraction_time_s is not None and record.contraction_time_s > 0.0
    assert record.final_terms == 2
    assert record.peak_terms == 2
    assert record.expectation_value == pytest.approx(COS_THETA, abs=1e-15)
    assert record.absolute_error == pytest.approx(0.0, abs=1e-15)
    assert record.peak_memory_kb is None or record.peak_memory_kb > 0
    assert record.extra["theta"] == THETA
    assert record.extra["state"] == "z+"
    assert "baseline_peak_memory_kb" in record.extra
    assert "peak_memory_kb_delta" in record.extra

    provenance = record.provenance
    assert provenance.date != ""
    assert provenance.hostname != ""
    assert provenance.seeds == {"circuit": 7}
    assert provenance.library_versions.get("paulistrings") is not None

    # Serializable end to end, so it can go straight into write_results.
    assert report.RunRecord.from_dict(record.to_dict()).final_terms == 2


def test_run_propagation_without_contraction_leaves_value_fields_none():
    record = harness.run_propagation(_circuit(), _observable(), None, "forward")
    assert record.contraction_time_s is None
    assert record.expectation_value is None
    assert record.absolute_error is None
    assert record.final_terms == 2
    assert "state" not in record.extra


def test_run_propagation_accepts_a_contract_callable():
    reference = _evolved()
    record = harness.run_propagation(
        _circuit(),
        _observable(),
        None,
        "heisenberg",
        contract=lambda evolved: evolved.overlap(reference),
    )
    # <O,O> = sum |c_i|^2 = cos^2 + sin^2 = 1.
    assert record.expectation_value == pytest.approx(1.0, abs=1e-15)
    assert record.contraction_time_s is not None


def test_run_propagation_accepts_a_ready_made_policy_object():
    record = harness.run_propagation(
        _circuit(),
        _observable(),
        truncation.weight(1) & truncation.coeff(0.5),
        "heisenberg",
        state="z+",
    )
    assert record.final_terms == 1
    # Opaque policy: labelled by repr, so it cannot feed a truncation-keyed plot.
    assert record.truncation == {"policy": "Truncation(And(Weight(1), Coeff(0.5)))"}


def test_run_propagation_does_not_consume_the_observable():
    observable = _observable()
    first = harness.run_propagation(_circuit(), observable, None, "heisenberg", state="z+")
    second = harness.run_propagation(_circuit(), observable, None, "heisenberg", state="z+")
    assert len(observable) == 1
    assert first.expectation_value == second.expectation_value


def test_run_propagation_requires_an_explicit_valid_direction():
    with pytest.raises(TypeError):
        harness.run_propagation(_circuit(), _observable(), None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="direction"):
        harness.run_propagation(_circuit(), _observable(), None, "backwards")


def test_run_propagation_rejects_state_and_contract_together():
    with pytest.raises(ValueError, match="state= or contract="):
        harness.run_propagation(
            _circuit(),
            _observable(),
            None,
            "heisenberg",
            state="z+",
            contract=lambda s: 0.0,
        )


def test_run_propagation_asserts_the_pin_when_threads_is_one():
    if _is_pinned():
        record = harness.run_propagation(
            _circuit(), _observable(), None, "heisenberg", state="z+", threads=1
        )
        assert record.provenance.thread_count == 1
    else:
        with pytest.raises(RuntimeError, match="RAYON_NUM_THREADS"):
            harness.run_propagation(
                _circuit(), _observable(), None, "heisenberg", state="z+", threads=1
            )


def test_run_propagation_records_the_worker_estimate_when_threads_unset():
    record = harness.run_propagation(_circuit(), _observable(), None, "heisenberg")
    estimate = harness.rayon_worker_estimate()
    assert record.provenance.thread_count == estimate
    if estimate is not None:
        assert "observed_threads" in record.extra


def test_run_propagation_warmup_flag_does_not_change_the_result():
    warm = harness.run_propagation(
        _circuit(), _observable(), None, "heisenberg", state="z+", warmup=True
    )
    cold = harness.run_propagation(
        _circuit(), _observable(), None, "heisenberg", state="z+", warmup=False
    )
    assert warm.final_terms == cold.final_terms
    assert warm.expectation_value == pytest.approx(cold.expectation_value, abs=1e-15)


# --- diff_pauli_sums / parity gate --------------------------------------


def test_diff_identical_sums_matches():
    diff = harness.diff_pauli_sums(_evolved(), _evolved())
    assert diff.is_match
    assert diff.matched == 2 == diff.terms_a == diff.terms_b
    assert diff.max_abs_delta == 0.0
    assert diff.describe().startswith("2 vs 2 terms")


def test_diff_catches_an_injected_coefficient_difference():
    reference = _evolved()
    perturbed = _observable().propagate(
        _circuit_with_angle(THETA + 1e-6), None, direction="heisenberg"
    )
    diff = harness.diff_pauli_sums(reference, perturbed, tol=1e-12)
    assert not diff.is_match
    assert diff.matched == 2
    assert not diff.only_in_a and not diff.only_in_b
    # d(sin theta)/dtheta = cos theta ~ 0.878, so the shift is ~8.8e-7.
    assert diff.max_abs_delta == pytest.approx(1e-6 * COS_THETA, rel=1e-3)
    assert diff.max_delta_key is not None
    assert "max |Δcoeff|" in diff.describe()

    # ... and a tolerance wide enough to cover it makes the same pair match.
    assert harness.diff_pauli_sums(reference, perturbed, tol=1e-5).is_match


def test_diff_catches_a_missing_key():
    reference = _evolved()
    truncated = _evolved(harness.make_policy(min_abs_coeff=0.5))
    diff = harness.diff_pauli_sums(reference, truncated)
    assert not diff.is_match
    assert diff.matched == 1
    assert len(diff.only_in_a) == 1
    assert not diff.only_in_b
    key, coeff = diff.only_in_a[0]
    assert coeff.real == pytest.approx(SIN_THETA, abs=1e-15)
    # Y on qubit 0 is the symplectic key (x=1, z=1).
    assert key == ((1,), (1,))
    assert "only in A" in diff.describe()


def test_diff_tolerates_a_negligible_key_present_on_one_side_only():
    # A term that only just survived one engine's cutoff, with a coefficient
    # inside the tolerance, is agreement to floating point, not a divergence.
    reference = _evolved()
    diff = harness.diff_pauli_sums(reference, _evolved(), tol=1e-12)
    assert diff.is_match
    loose = harness.diff_pauli_sums(
        reference, _evolved(harness.make_policy(min_abs_coeff=0.5)), tol=1.0
    )
    assert loose.is_match


def test_diff_rejects_mismatched_qubit_counts():
    other = PauliSum.from_strings({"ZII": 1.0}, num_qubits=3)
    with pytest.raises(ValueError, match="different qubit counts"):
        harness.diff_pauli_sums(_evolved(), other)


def _circuit_with_angle(theta: float) -> Circuit:
    c = Circuit(NUM_QUBITS)
    c.rx(theta, 0)
    return c


def _record(**overrides) -> report.RunRecord:
    base = harness.run_propagation(
        _circuit(), _observable(), (None, 1e-9), "heisenberg", state="z+"
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_check_term_parity_passes_on_matched_runs():
    result = harness.check_term_parity(_record(), _record())
    assert result.ok
    assert result.reasons == []
    assert result.describe() == "parity holds"


def test_check_term_parity_catches_each_mismatch_class():
    reference = _record()

    assert not harness.check_term_parity(reference, _record(final_terms=3)).ok
    assert not harness.check_term_parity(reference, _record(peak_terms=99)).ok
    assert not harness.check_term_parity(reference, _record(n_qubits=4)).ok
    assert not harness.check_term_parity(reference, _record(direction="forward")).ok
    assert not harness.check_term_parity(
        reference, _record(truncation={"min_abs_coeff": 1e-8})
    ).ok
    assert not harness.check_term_parity(
        reference, _record(expectation_value=COS_THETA + 1e-6)
    ).ok
    # Within tolerance is not a mismatch.
    assert harness.check_term_parity(
        reference, _record(expectation_value=COS_THETA + 1e-15)
    ).ok


def test_check_term_parity_reasons_name_the_engines():
    other = _record(engine="PauliPropagation.jl", final_terms=5)
    result = harness.check_term_parity(_record(), other)
    assert not result.ok
    joined = result.describe()
    assert "final_terms differ" in joined
    assert "PauliPropagation.jl" in joined


def test_require_parity_dispatches_on_records_and_sums():
    assert harness.require_parity(_record(), _record()).ok
    assert harness.require_parity(_evolved(), _evolved()).is_match

    with pytest.raises(harness.ParityError, match="final_terms differ"):
        harness.require_parity(_record(), _record(final_terms=17))
    with pytest.raises(harness.ParityError, match="evolved sums diverge"):
        harness.require_parity(_evolved(), _evolved(harness.make_policy(min_abs_coeff=0.5)))
    with pytest.raises(TypeError, match="two RunRecords or two Pauli sums"):
        harness.require_parity(_record(), _evolved())


def test_require_parity_label_appears_in_the_dump():
    with pytest.raises(harness.ParityError, match=r"\[A/jl n=127\]"):
        harness.require_parity(_record(), _record(final_terms=17), label="A/jl n=127")


# --- convergence_sweep --------------------------------------------------

GRID = [(None, 0.9), (None, 0.5), (None, 0.1)]


def _build_run(spec: harness.TruncationSpec) -> report.RunRecord:
    return harness.run_propagation(
        _circuit(), _observable(), spec, "heisenberg", state="z+"
    )


def test_convergence_sweep_returns_one_record_per_grid_point_in_order():
    records = harness.convergence_sweep(_build_run, GRID)
    assert [r.truncation for r in records] == [
        {"min_abs_coeff": 0.9},
        {"min_abs_coeff": 0.5},
        {"min_abs_coeff": 0.1},
    ]
    # The hand-computed regimes from the module docstring.
    assert [r.final_terms for r in records] == [0, 1, 2]
    assert [r.expectation_value for r in records] == pytest.approx(
        [0.0, COS_THETA, COS_THETA], abs=1e-15
    )
    assert all(r.absolute_error is None for r in records)


def test_convergence_sweep_backfills_error_from_an_oracle():
    records = harness.convergence_sweep(_build_run, GRID, oracle_value=COS_THETA)
    assert [r.absolute_error for r in records] == pytest.approx(
        [COS_THETA, 0.0, 0.0], abs=1e-15
    )


def test_convergence_sweep_records_feed_the_convergence_panel():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    records = harness.convergence_sweep(_build_run, GRID, oracle_value=COS_THETA)
    figure = report.plot_convergence_panel(records, reference_value=COS_THETA)
    # One engine curve plus the reference line.
    assert len(figure.axes[0].lines) == 2


def test_convergence_sweep_rejects_an_empty_grid():
    with pytest.raises(ValueError, match="empty"):
        harness.convergence_sweep(_build_run, [])


def test_convergence_sweep_catches_a_build_run_that_ignores_its_spec():
    def ignores_spec(spec):
        return harness.run_propagation(
            _circuit(), _observable(), (None, 1e-9), "heisenberg", state="z+"
        )

    with pytest.raises(ValueError, match="ignored its spec"):
        harness.convergence_sweep(ignores_spec, GRID)


def test_convergence_sweep_rejects_a_non_record_return():
    with pytest.raises(TypeError, match="RunRecord"):
        harness.convergence_sweep(lambda spec: "nope", GRID)


# --- time_to_accuracy ---------------------------------------------------


def test_time_to_accuracy_selects_the_hand_predictable_grid_point():
    result = harness.time_to_accuracy(_build_run, COS_THETA, 1e-9, GRID)

    assert len(result.records) == 3
    assert result.achieved
    # eps=0.9 wipes the operator out (error = cos theta); eps=0.5 is already
    # exact because the dropped Y term contributes 0 to <Z> in |0...0>.
    assert result.first_index == 1
    assert result.first_spec == harness.TruncationSpec(None, 0.5)
    assert result.first is result.records[1]
    assert result.first.absolute_error == pytest.approx(0.0, abs=1e-15)
    # Both passing points are exact, so "cheapest" is decided by wall time and
    # may be either of them — but never the failing one.
    assert result.cheapest_index in (1, 2)
    assert result.cheapest is result.records[result.cheapest_index]
    assert "first pass in grid order" in result.describe()


def test_time_to_accuracy_reports_no_selection_when_nothing_converges():
    result = harness.time_to_accuracy(_build_run, COS_THETA, 1e-9, [(None, 0.9)])
    assert not result.achieved
    assert result.first is None and result.cheapest is None
    assert result.first_spec is None and result.cheapest_spec is None
    assert len(result.records) == 1
    assert "no grid point met the bar" in result.describe()


def test_time_to_accuracy_runs_the_whole_grid_for_the_curve():
    result = harness.time_to_accuracy(_build_run, COS_THETA, 1e-9, GRID)
    assert len(result.records) == len(GRID) == len(result.specs)

    early = harness.time_to_accuracy(_build_run, COS_THETA, 1e-9, GRID, stop_early=True)
    assert len(early.records) == 2
    assert len(early.specs) == 2
    assert early.first_index == 1


def test_time_to_accuracy_epsilon_bar_is_strict():
    # Grid point 0's error is exactly cos(theta); an epsilon equal to it must
    # not count as a pass.
    exact = harness.time_to_accuracy(_build_run, COS_THETA, COS_THETA, [(None, 0.9)])
    assert not exact.achieved
    looser = harness.time_to_accuracy(
        _build_run, COS_THETA, COS_THETA * 1.001, [(None, 0.9)]
    )
    assert looser.achieved


def test_time_to_accuracy_rejects_unscorable_runs():
    def no_contraction(spec):
        return harness.run_propagation(_circuit(), _observable(), spec, "heisenberg")

    with pytest.raises(ValueError, match="no expectation value"):
        harness.time_to_accuracy(no_contraction, COS_THETA, 1e-9, GRID)


def test_time_to_accuracy_rejects_a_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        harness.time_to_accuracy(_build_run, COS_THETA, 0.0, GRID)


def test_time_to_accuracy_records_feed_the_error_vs_runtime_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    result = harness.time_to_accuracy(_build_run, COS_THETA, 1e-9, GRID)
    figure = report.plot_error_vs_runtime(result.records)
    # Only the non-zero-error point is plottable on log axes; the curve exists.
    assert len(figure.axes[0].lines) == 1


def test_time_to_accuracy_grid_accepts_a_weight_sweep():
    grid = [(0, None), (1, None)]
    result = harness.time_to_accuracy(_build_run, COS_THETA, 1e-9, grid)
    assert [r.truncation for r in result.records] == [{"max_weight": 0}, {"max_weight": 1}]
    # Weight 0 keeps nothing (no identity term), weight 1 keeps both terms.
    assert [r.final_terms for r in result.records] == [0, 2]
    assert result.first_index == 1
