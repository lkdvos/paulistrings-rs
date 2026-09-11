"""Showcase B6 -- CI-safe correctness gates (adapted plan §6 Part B "B6").

`examples/b6_resource_probes/run_b6.py` is the full showcase (narrative,
sweeps, committed figures); this file pins the properties of
`examples/b6_resource_probes/resource_probes.py` that must never regress:

* the array-based diagnostics agree with **independent dense oracles** (a
  brute-force sweep over all `4^n` Pauli traces; a reshape-and-SVD of the
  dense matrix) to `1e-10` at `n = 4` and `n = 6`;
* both diagnostics read **zero at the kicked-Ising Clifford points**, exactly
  at `theta_h = 0` and to floating-point dust at `theta_h = pi/2`;
* the algebraic identities that define the quantities (`S_2 = -ln(1 - L)`,
  Rényi monotonicity in α, `sum_k s_k^2 = sum_P |c_P|^2`, additivity over a
  tensor product, the unnormalized-OSE offset) hold on real evolved sums;
* the size guards actually refuse, rather than allocating.

Deliberately small (`n <= 6`, depth <= 3) so the whole file runs in a couple
of seconds and needs nothing beyond `paulistrings` + numpy --
`examples/common/{circuits,observables,oracles}.py` and `resource_probes.py`
are numpy-only at import time, so this is CI-visible with no `importorskip`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from paulistrings import PauliSum, truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_B6_DIR = _EXAMPLES_DIR / "b6_resource_probes"
for _path in (str(_EXAMPLES_DIR), str(_B6_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import resource_probes as probes  # noqa: E402

from common import circuits, observables, oracles  # noqa: E402

TOL = 1e-10
#: `theta_h = pi/2` is a Clifford angle only in exact arithmetic: `cos(pi/2)`
#: is 6.1e-17, not 0, so the branch that should cancel survives as dust and the
#: sum keeps thousands of terms with coefficients around 1e-49. Both
#: diagnostics are quadratic in those, hence a bound rather than an equality.
CLIFFORD_DUST_BOUND = 1e-25

_DIAGNOSTICS = (
    "pauli_renyi2",
    "pauli_linear",
    "pauli_shannon",
    "op_entanglement",
    "op_entanglement_renyi2",
)


def _chain(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _evolve(n: int, steps: int, theta_h: float, *, policy=None):
    """Heisenberg-evolve `Z_{n/2}` through a 1D kicked-Ising Trotter circuit."""
    circuit = circuits.heavy_hex_kicked_ising(n, steps, theta_h, edges=_chain(n))
    return observables.single_z(n // 2, n).propagate(
        circuit, policy, direction="heisenberg"
    )


def _diagnostics(pauli_sum, cut: int) -> dict[str, float]:
    spectrum = probes.operator_schmidt_spectrum(pauli_sum, cut)
    return {
        "pauli_renyi2": probes.pauli_spectrum_renyi2(pauli_sum),
        "pauli_linear": probes.pauli_spectrum_linear(pauli_sum),
        "pauli_shannon": probes.pauli_spectrum_shannon(pauli_sum),
        "op_entanglement": probes.renyi_entropy(spectrum, 1.0),
        "op_entanglement_renyi2": probes.renyi_entropy(spectrum, 2.0),
    }


# ---------------------------------------------------------------------------
# The dense cross-check -- the gate that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "steps"), [(4, 2), (6, 3)])
def test_diagnostics_match_independent_dense_oracles(n, steps):
    """Both diagnostic families, recomputed from a dense `numpy.kron` matrix.

    The dense route shares nothing with the array-based probes but the numpy
    export itself: the Pauli spectrum is rebuilt from all `4^n` traces
    `tr(P O) / 2^n`, and the operator Schmidt spectrum from a reshape of the
    dense matrix that never looks at a Pauli label or a symplectic bit.
    """
    cut = n // 2
    evolved = _evolve(n, steps, 0.6)
    dense = probes.dense_matrix(oracles.pauli_terms(evolved))

    # sum_P |c_P|^2 == tr(O^dag O) / 2^n, the identity that makes the spectral
    # weight a property of the operator and not of its representation.
    sparse_weight = probes.hilbert_schmidt_weight(evolved)
    dense_weight = float(np.trace(dense.conj().T @ dense).real) / (1 << n)
    assert abs(sparse_weight - dense_weight) <= TOL
    # Untruncated unitary Heisenberg evolution of a unit Pauli string is an
    # orthogonal rotation of the coefficient vector.
    assert abs(sparse_weight - 1.0) <= TOL

    p_dense = probes.dense_pauli_spectrum_probabilities(dense, n)
    assert abs(probes.pauli_spectrum_renyi2(evolved) - probes.renyi_entropy(p_dense, 2.0)) <= TOL
    assert abs(probes.pauli_spectrum_shannon(evolved) - probes.renyi_entropy(p_dense, 1.0)) <= TOL
    assert abs(probes.pauli_spectrum_linear(evolved) - (1.0 - (p_dense**2).sum())) <= TOL

    lambdas_dense = probes.dense_operator_schmidt_spectrum(dense, n, cut)
    lambdas_sparse = probes.operator_schmidt_spectrum(evolved, cut)
    for alpha in (1.0, 2.0):
        gap = abs(
            probes.renyi_entropy(lambdas_sparse, alpha)
            - probes.renyi_entropy(lambdas_dense, alpha)
        )
        assert gap <= TOL, f"alpha={alpha} operator entanglement gap {gap:.3e}"


def test_dense_cross_check_is_sensitive_to_the_cut():
    """A different bipartition really does give a different dense answer.

    Guards against a cross-check that would pass for the wrong reason -- e.g.
    if both routes silently ignored `cut`, every bipartition would agree with
    every other and the test above would be vacuous.
    """
    n, steps = 6, 3
    evolved = _evolve(n, steps, 0.6)
    dense = probes.dense_matrix(oracles.pauli_terms(evolved))
    values = {}
    for cut in (1, 2, 3):
        sparse = probes.operator_entanglement_entropy(evolved, cut)
        oracle = probes.renyi_entropy(
            probes.dense_operator_schmidt_spectrum(dense, n, cut), 1.0
        )
        assert abs(sparse - oracle) <= TOL
        values[cut] = sparse
    assert len(set(round(v, 9) for v in values.values())) == len(values), (
        f"the three cuts must give three different entropies, got {values}"
    )


# ---------------------------------------------------------------------------
# Clifford points
# ---------------------------------------------------------------------------


def test_a_single_pauli_string_has_no_resource_at_all():
    observable = observables.single_z(3, 8)
    for key, value in _diagnostics(observable, 4).items():
        assert value == 0.0, f"{key} must be exactly 0 for a one-term sum, got {value!r}"
    assert probes.hilbert_schmidt_weight(observable) == 1.0
    assert probes.schmidt_matrix_shape(observable, 4) == (1, 1)


def test_clifford_evolution_leaves_both_diagnostics_at_zero():
    """`theta_h = 0` (identity X layer) and `theta_h = pi/2` (Clifford quarter
    turn) both keep a single-Pauli seed a single Pauli string, so every
    diagnostic must vanish -- exactly at 0, and to dust at pi/2.
    """
    for theta_h, bound in ((0.0, 0.0), (math.pi / 2, CLIFFORD_DUST_BOUND)):
        evolved = _evolve(8, 3, theta_h)
        for key, value in _diagnostics(evolved, 4).items():
            assert abs(value) <= bound, (
                f"theta_h={theta_h} is a Clifford point, so {key} must vanish; "
                f"got {value:.3e} > {bound:.0e}"
            )


def test_a_generic_kick_angle_raises_both_diagnostics():
    evolved = _evolve(8, 3, 0.6)
    for key, value in _diagnostics(evolved, 4).items():
        assert value > 1e-3, f"{key} should be well above zero off the Clifford points"


# ---------------------------------------------------------------------------
# Defining identities
# ---------------------------------------------------------------------------


def test_renyi2_and_the_linear_variant_are_the_same_information():
    evolved = _evolve(8, 3, 0.6)
    renyi2 = probes.pauli_spectrum_renyi2(evolved)
    linear = probes.pauli_spectrum_linear(evolved)
    assert abs(renyi2 + math.log1p(-linear)) <= 1e-12


def test_renyi_entropy_is_non_increasing_in_alpha():
    evolved = _evolve(8, 3, 0.6)
    p = probes.pauli_spectrum_probabilities(evolved)
    values = [probes.renyi_entropy(p, alpha) for alpha in (0.5, 1.0, 2.0, 4.0, np.inf)]
    assert values == sorted(values, reverse=True), values


def test_renyi_entropy_hand_computed():
    """`p = (1/2, 1/4, 1/4)`: hand values, not another function's output."""
    p = np.array([0.5, 0.25, 0.25])
    # sum p^2 = 1/4 + 1/16 + 1/16 = 3/8
    assert probes.renyi_entropy(p, 2.0) == pytest.approx(-math.log(3 / 8), abs=1e-15)
    # -sum p ln p = ln 2 / 2 + 2 * (ln 4 / 4) = (3/2) ln 2
    assert probes.renyi_entropy(p, 1.0) == pytest.approx(1.5 * math.log(2), abs=1e-15)
    # min-entropy = -ln max p = ln 2
    assert probes.renyi_entropy(p, np.inf) == pytest.approx(math.log(2), abs=1e-15)
    # A point distribution has zero entropy at every alpha.
    for alpha in (0.5, 1.0, 2.0, np.inf):
        assert probes.renyi_entropy(np.array([1.0, 0.0, 0.0]), alpha) == 0.0


def test_schmidt_singular_values_carry_the_hilbert_schmidt_weight():
    """`sum_k s_k^2 = ||M||_F^2 = sum_P |c_P|^2` -- the Frobenius identity that
    makes the operator Schmidt matrix a faithful repackaging of the
    coefficient vector, so no weight is created or lost by the reshape.
    """
    for policy in (None, truncation.coeff(1e-3)):
        evolved = _evolve(10, 3, 0.6, policy=policy)
        values = probes.operator_schmidt_values(evolved, 5)
        assert float((values**2).sum()) == pytest.approx(
            probes.hilbert_schmidt_weight(evolved), rel=1e-12
        )


def test_a_factorized_operator_has_zero_operator_entanglement():
    """`O = (X_0 + 2 Z_0) ⊗ Y_2` on 4 qubits: a rank-1 Schmidt matrix across
    the cut at 2, hence zero operator entanglement, while its *Pauli* spectrum
    is spread over two strings and so is not zero. The two diagnostics measure
    different things, and this is the cheapest witness of that.
    """
    observable = PauliSum.from_strings({"XIYI": 1.0, "ZIYI": 2.0}, num_qubits=4)
    assert probes.schmidt_matrix_shape(observable, 2) == (2, 1)
    assert probes.operator_entanglement_entropy(observable, 2) == pytest.approx(0.0, abs=1e-14)
    # p = (1/5, 4/5), so sum p^2 = 17/25 and S_2 = -ln(17/25) = 0.3857...
    assert probes.pauli_spectrum_renyi2(observable) == pytest.approx(
        -math.log(17 / 25), abs=1e-12
    )


def test_the_pauli_spectrum_entropy_is_additive_over_a_tensor_product():
    """`S_α(A ⊗ B) = S_α(A) + S_α(B)`, from `p` factorizing.

    Built by hand: `A = X_0 + 2 Y_0` on qubit 0 and `B = Z_2 + 3 X_2` on qubit
    2 of a 4-qubit register, so the product is the 4-term sum below with the
    coefficient products.
    """
    n = 4
    left = PauliSum.from_strings({"XIII": 1.0, "YIII": 2.0}, num_qubits=n)
    right = PauliSum.from_strings({"IIZI": 1.0, "IIXI": 3.0}, num_qubits=n)
    product = PauliSum.from_strings(
        {"XIZI": 1.0, "XIXI": 3.0, "YIZI": 2.0, "YIXI": 6.0}, num_qubits=n
    )
    for alpha in (0.5, 1.0, 2.0, 3.0):
        expected = probes.pauli_spectrum_renyi(left, alpha) + probes.pauli_spectrum_renyi(
            right, alpha
        )
        assert probes.pauli_spectrum_renyi(product, alpha) == pytest.approx(
            expected, abs=1e-12
        )


def test_the_unnormalized_ose_differs_only_by_the_weight_term():
    """arXiv:2510.22311's literal `S^α(O)` vs this module's renormalized form.

    They agree exactly when `sum|c|^2 = 1` and differ by
    `(α/(1-α)) ln(sum|c|^2)` when truncation has thrown weight away -- the
    reason the showcase plots the renormalized one.
    """
    exact = _evolve(10, 3, 0.6)
    assert probes.hilbert_schmidt_weight(exact) == pytest.approx(1.0, abs=1e-12)
    assert probes.pauli_spectrum_renyi_unnormalized(exact, 2.0) == pytest.approx(
        probes.pauli_spectrum_renyi(exact, 2.0), abs=1e-12
    )

    truncated = _evolve(10, 3, 0.6, policy=truncation.coeff(1e-2))
    weight = probes.hilbert_schmidt_weight(truncated)
    assert weight < 1.0 - 1e-9, "the cutoff must actually discard weight for this test"
    for alpha in (2.0, 3.0):
        offset = alpha / (1.0 - alpha) * math.log(weight)
        assert probes.pauli_spectrum_renyi_unnormalized(truncated, alpha) == pytest.approx(
            probes.pauli_spectrum_renyi(truncated, alpha) + offset, abs=1e-12
        )


# ---------------------------------------------------------------------------
# Guards and error paths
# ---------------------------------------------------------------------------


def test_the_schmidt_matrix_size_guard_refuses_instead_of_allocating():
    evolved = _evolve(10, 3, 0.6)
    rows, cols = probes.schmidt_matrix_shape(evolved, 5)
    assert rows * cols > 1
    with pytest.raises(ValueError, match="over the .* guard"):
        probes.operator_schmidt_values(evolved, 5, max_entries=1)
    # The same call succeeds once the guard is raised past the real size.
    probes.operator_schmidt_values(evolved, 5, max_entries=rows * cols)


def test_the_dense_spectrum_guard_refuses_a_hopeless_size():
    dense = np.eye(1 << 10, dtype=np.complex128)
    with pytest.raises(ValueError, match="past the n<=8 guard"):
        probes.dense_pauli_spectrum_probabilities(dense, 10)


def test_a_trivial_bipartition_is_rejected():
    evolved = _evolve(8, 2, 0.6)
    for cut in (0, 8, -1, 9):
        with pytest.raises(ValueError, match="bipartition boundary"):
            probes.operator_entanglement_entropy(evolved, cut)


def test_an_empty_sum_is_rejected_rather_than_reported_as_zero():
    """A cutoff above every coefficient leaves nothing, and a Pauli spectrum
    over no terms does not exist -- reporting `0` there would look exactly like
    the Clifford answer, which is the one confusion worth being loud about.
    """
    empty = _evolve(8, 1, 0.6, policy=truncation.coeff(2.0))
    assert len(empty) == 0
    with pytest.raises(ValueError, match="empty"):
        probes.pauli_spectrum_renyi2(empty)
    with pytest.raises(ValueError, match="empty"):
        probes.hilbert_schmidt_weight(empty)
    with pytest.raises(ValueError, match="empty"):
        probes.operator_entanglement_entropy(empty, 4)


def test_renyi_entropy_rejects_an_unnormalized_input():
    with pytest.raises(ValueError, match="sum to 1"):
        probes.renyi_entropy(np.array([0.5, 0.25]), 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        probes.renyi_entropy(np.array([1.5, -0.5]), 2.0)
    with pytest.raises(ValueError, match="empty distribution"):
        probes.renyi_entropy(np.array([]), 2.0)
