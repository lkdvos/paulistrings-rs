"""Benchmark D correctness gate -- 1D Trotterized XXZ chain (plan §6 Part A, D).

CI-safe half of benchmark D; the measured curves live in
`examples/xxz_chain/run_benchmark_d.py` (driver) and `examples/xxz_chain/README.md`
(narrative + committed figures). Everything here is small and fast: the
statevector cross-checks run at `n = 12`, and the growth-law check at `n = 20`
with at most four Trotter steps.

Two things are gated:

1. **Statevector agreement.** The Heisenberg-picture propagation of a central
   `Z_c` and of a weight-2 `Z_c Z_{c+1}`, contracted against a domain-wall
   product state, equals `oracles.statevector_expectation` at tight truncation
   -- for the free (`Jz = 0`) *and* interacting (`Jz = 0.5`) regime, at several
   Trotter depths. Needs qiskit + Aer, so it `importorskip`s them; CI's python
   job installs numpy only.
2. **The `Jz = 0` quadratic growth law.** With no truncation, the number of
   non-zero Pauli terms produced by a weight-1 seed grows as `s^2` in the
   number of Trotter steps `s`, until the light cone reaches the chain
   boundary. Why: at `Jz = 0` the XX+YY chain is free -- Jordan-Wigner turns it
   into hopping Majoranas, every Trotter gate is Gaussian, and a single
   `Z_c = -i g_{2c} g_{2c+1}` therefore stays a sum of Majorana *bilinears*
   `g_a g_b`, each of which is exactly one Pauli string. The reachable set of
   `a`, `b` is the light cone, which widens by a fixed number of sites per
   step, so the count is (cone width)^2 = O(s^2).

   The pinned quantity is the **log-log slope**, not the counts: the exact
   integer identity the measurement shows (`terms(s) == 16 s^2`) is recorded in
   `examples/xxz_chain/README.md` as an observation, and pinning it here would
   turn a Trotter-decomposition detail (how far one step's bond sweep pushes
   support) into a test of the engine.

   Only numpy is needed, so this part always runs in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_DRIVER_DIR = _EXAMPLES_DIR / "xxz_chain"
for _p in (_EXAMPLES_DIR, _DRIVER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_benchmark_d as driver
from common import circuits, observables

DT = driver.DT
DIRECTION = driver.DIRECTION

#: Small enough for a dense reference to be instant.
CHECK_N = 12
#: Big enough that four Trotter steps stay inside the light cone
#: (`driver.unsaturated_max_steps(20) == 4`).
GROWTH_N = 20

TOL = 1e-9


def _observables(n):
    c = driver.center(n)
    return {
        "Z_c": observables.single_z(c, n),
        "Z_c Z_c+1": observables.pauli_sum_from_support({c: "Z", c + 1: "Z"}, n),
    }


# --- 1. Statevector agreement ------------------------------------------


@pytest.fixture(scope="module")
def oracles():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from common import oracles as _oracles

    return _oracles


@pytest.mark.parametrize("Jz", [driver.JZ_FREE, driver.JZ_INTERACTING])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_matches_statevector(oracles, Jz, steps):
    """Both observables, both regimes, three depths: exact to ~1e-15."""
    from paulistrings import truncation

    n = CHECK_N
    state = driver.domain_wall_state(n)
    spec = oracles.record_gates(circuits.xxz_chain_trotter, n, steps, Jz=Jz, dt=DT)
    circuit = spec.to_circuit()
    policy = truncation.coeff(driver.TIGHT_EPS)
    for name, obs in _observables(n).items():
        exact = oracles.statevector_expectation(spec, obs, state)
        got = complex(
            obs.propagate(circuit, policy, direction=DIRECTION).expectation(state)
        )
        assert abs(got - exact) < TOL, (
            f"Jz={Jz} steps={steps} {name}: paulistrings={got!r} "
            f"statevector={exact!r}"
        )


def test_domain_wall_state_is_not_an_eigenstate(oracles):
    """Guard against a vacuously-passing reference.

    `|0...0>` is an eigenstate of the XXZ Hamiltonian, so every expectation
    would be a time-independent constant and the agreement test above would
    hold no matter what the engine did. This asserts the domain wall actually
    evolves (and that the initial value is the exact `-1` the state prescribes,
    since qubit `c = n//2` is in `|1>`).
    """
    n = CHECK_N
    state = driver.domain_wall_state(n)
    obs = observables.single_z(driver.center(n), n)
    zero_steps = circuits.xxz_chain_trotter(n, 0, Jz=driver.JZ_INTERACTING, dt=DT)
    initial = complex(
        obs.propagate(zero_steps, None, direction=DIRECTION).expectation(state)
    ).real
    assert initial == pytest.approx(-1.0, abs=1e-12)

    evolved = circuits.xxz_chain_trotter(n, 3, Jz=driver.JZ_INTERACTING, dt=DT)
    later = complex(
        obs.propagate(evolved, None, direction=DIRECTION).expectation(state)
    ).real
    assert abs(later - initial) > 1e-3, "the domain wall did not melt at all"


# --- 2. The Jz = 0 quadratic growth law --------------------------------


def _untruncated_counts(n, max_steps, Jz, obs):
    counts = []
    for s in range(1, max_steps + 1):
        circuit = circuits.xxz_chain_trotter(n, s, Jz=Jz, dt=DT)
        counts.append(len(obs.propagate(circuit, None, direction=DIRECTION)))
    return counts


def test_free_regime_term_count_grows_quadratically():
    """Log-log slope of terms-vs-steps is 2 at `Jz = 0` (free fermions)."""
    n = GROWTH_N
    max_steps = driver.unsaturated_max_steps(n)
    assert max_steps >= 3, "need at least three unsaturated points for a fit"
    steps = list(range(1, max_steps + 1))
    counts = _untruncated_counts(
        n, max_steps, driver.JZ_FREE, observables.single_z(driver.center(n), n)
    )
    slope = driver.loglog_slope(steps, counts)
    assert slope == pytest.approx(2.0, abs=driver.GROWTH_SLOPE_TOL), (
        f"expected a quadratic growth law at Jz=0, got log-log slope {slope:.4f} "
        f"from counts {counts} at steps {steps}"
    )


def test_growth_law_fit_window_excludes_the_boundary():
    """The fit window is the light cone's, not an arbitrary cut.

    Past `unsaturated_max_steps` the cone has reached the chain ends and the
    count is boundary-limited, so the slope drops below 2. Pinning that the
    *excluded* region really does break the law is what makes the exclusion a
    physical statement rather than a convenient one.
    """
    n = GROWTH_N
    beyond = 2 * driver.unsaturated_max_steps(n)
    counts = _untruncated_counts(
        n, beyond, driver.JZ_FREE, observables.single_z(driver.center(n), n)
    )
    saturated_slope = driver.loglog_slope(
        list(range(driver.unsaturated_max_steps(n), beyond + 1)),
        counts[driver.unsaturated_max_steps(n) - 1:],
    )
    assert saturated_slope < 2.0 - driver.GROWTH_SLOPE_TOL, (
        f"the boundary-limited tail still looks quadratic (slope "
        f"{saturated_slope:.4f}); counts {counts}"
    )


def test_interacting_regime_grows_faster_than_quadratically():
    """`Jz = 0.5` breaks the free-fermion structure, and the count explodes.

    Two steps is enough to make the point cheaply: quadratic growth would give
    a 4x increase from `s = 1` to `s = 2`, and the measured factor is ~240.
    """
    n = GROWTH_N
    obs = observables.single_z(driver.center(n), n)
    free = _untruncated_counts(n, 2, driver.JZ_FREE, obs)
    interacting = _untruncated_counts(n, 2, driver.JZ_INTERACTING, obs)
    assert free[1] / free[0] == pytest.approx(4.0, rel=0.01)
    assert interacting[1] / interacting[0] > 50.0, (
        f"expected super-quadratic growth at Jz=0.5, got counts {interacting}"
    )
    assert interacting[0] > free[0], "the ZZ rotation added no terms at all"


def test_free_regime_growth_is_independent_of_dt_and_seed_site():
    """The law is structural (light-cone combinatorics), not numerical.

    Changing the Trotter step size changes every coefficient and changes no
    count; moving the seed off center changes neither, as long as the cone
    still fits. A cancellation-sensitive count would fail this.
    """
    n = 40
    steps = 4
    base = _untruncated_counts(
        n, steps, driver.JZ_FREE, observables.single_z(driver.center(n), n)
    )
    assert base == [16 * s * s for s in range(1, steps + 1)]

    for dt in (0.05, 0.37):
        counts = []
        obs = observables.single_z(driver.center(n), n)
        for s in range(1, steps + 1):
            circuit = circuits.xxz_chain_trotter(n, s, Jz=driver.JZ_FREE, dt=dt)
            counts.append(len(obs.propagate(circuit, None, direction=DIRECTION)))
        assert counts == base, f"dt={dt} changed the counts: {counts} != {base}"

    off_center = _untruncated_counts(
        n, steps, driver.JZ_FREE, observables.single_z(12, n)
    )
    assert off_center == base
