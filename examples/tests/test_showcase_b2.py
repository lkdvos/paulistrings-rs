"""Showcase B2 -- CI-safe correctness gates (adapted plan §6 Part B "B2").

`examples/b2_noisy_verification/run_b2.py` is the showcase (it produces the
committed figures, `results.json` / `summary.json` and every narrative number at
`n = 127`); this file pins the properties that narrative rests on, at sizes that
run in seconds.

The load-bearing gate is an **independent dense noisy reference**, hand-rolled
here in numpy: the circuit's channels are applied as explicit Kraus operators to
a `2^n x 2^n` density matrix, forward in time from `|0...0><0...0|`, and the
answer is `Tr(O rho)`. That path shares no code with the engine and none with
`examples/common/oracles.py` -- whose statevector and stabilizer oracles both
*refuse* noise channels by design ("a density-matrix reference, which this module
does not provide"). It therefore checks the thing that could not otherwise be
checked: that `depolarize`, `dephase`, `amplitude_damping`, `pauli_channel` and
`depolarize2` implement the channels they claim to, in both directions, mixed
with Cliffords and non-Clifford rotations.

What is gated here
------------------
1. the dense cross-check above, per channel kind, in the **Heisenberg**
   direction (`Tr(O rho)`) and -- for the exhaustive version -- in the
   **forward** direction, where every one of the `4^n` Pauli coefficients of the
   evolved `rho` is compared, so a *missing* term fails too;
2. `amplitude_damping` specifically, since it is the only channel here that is
   neither self-adjoint nor key-preserving, and its `apply`/`apply_adjoint`
   orientation was wrong until commit e42095c;
3. the showcase's noise model: the `p = 0` leg is *identical* to the noiseless
   circuit, the `NOISELESS` model reproduces the shared builder, and the channel
   count per Trotter step is the documented `2n + 3|E|`;
4. the mechanism -- at the Clifford kick angle the evolved operator stays a
   single Pauli string whose coefficient is exactly `(1 - 4p/3)^hits` with `hits`
   hand-counted from the lattice, and at a generic angle a larger `p` keeps a
   strictly smaller tracked set at fixed cutoff;
5. the **citation**: `run_b2.claimable_references()` returns exactly Benchmark
   C's claimable rows, with the values this showcase quotes, read out of
   `benchmarks/python/deep_trotter/summary.json` rather than transcribed.

Everything here is numpy-only (no qiskit, no stim, no matplotlib), so it is
CI-visible with no `importorskip` (plan §4).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from paulistrings import PauliSum, truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from b2_noisy_verification import run_b2  # noqa: E402
from common import circuits, observables, oracles  # noqa: E402

TOLERANCE = 1e-10

# ==========================================================================
# The dense noisy reference -- hand-rolled, independent of the engine
# ==========================================================================

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    """A Pauli string as a dense matrix; character 0 is the most significant
    tensor factor, so label character `i` acts on qubit `i`."""
    out = np.ones((1, 1), dtype=complex)
    for ch in label:
        out = np.kron(out, _PAULI[ch])
    return out


def _embed_pauli(pauli: str, qubits, n: int) -> np.ndarray:
    label = ["I"] * n
    for ch, q in zip(pauli, qubits, strict=True):
        label[int(q)] = ch
    return _pauli_matrix("".join(label))


def _embed_1q(matrix: np.ndarray, q: int, n: int) -> np.ndarray:
    """A one-qubit matrix acting on qubit `q` of `n`, with qubit 0 most
    significant: `I_{2^q} (x) matrix (x) I_{2^(n-q-1)}`."""
    return np.kron(
        np.kron(np.eye(2**q, dtype=complex), matrix),
        np.eye(2 ** (n - q - 1), dtype=complex),
    )


def _gate_unitary(gate, n: int) -> np.ndarray:
    """The dense unitary of one gate object.

    Two-qubit Cliffords are written as sums of embedded Pauli products
    (`CNOT = (I + Z_c + X_t - Z_c X_t)/2`, `CZ = (I + Z_a + Z_b - Z_a Z_b)/2`,
    `SWAP = (II + XX + YY + ZZ)/2`) rather than as 4x4 blocks, so this reference
    never has to reason about tensor-factor ordering for a non-adjacent pair --
    the one place a dense cross-check is easy to get silently wrong.
    """
    name = str(gate["name"])
    qubits = [int(q) for q in gate["qubits"]]
    dim = 2**n
    identity = np.eye(dim, dtype=complex)

    if name in ("rx", "ry", "rz", "pauli_rotation"):
        pauli = name[1].upper() if name != "pauli_rotation" else str(gate["pauli"]).upper()
        theta = float(gate["theta"])
        p = _embed_pauli(pauli, qubits, n)
        return math.cos(theta / 2.0) * identity - 1j * math.sin(theta / 2.0) * p
    if name in ("x", "y", "z"):
        return _embed_pauli(name.upper(), qubits, n)
    if name == "h":
        return _embed_1q((_PAULI["X"] + _PAULI["Z"]) / math.sqrt(2.0), qubits[0], n)
    if name == "s":
        return _embed_1q(np.diag([1.0 + 0j, 1j]), qubits[0], n)
    if name == "cnot":
        control, target = qubits
        return 0.5 * (
            identity
            + _embed_pauli("Z", [control], n)
            + _embed_pauli("X", [target], n)
            - _embed_pauli("ZX", [control, target], n)
        )
    if name == "cz":
        a, b = qubits
        return 0.5 * (
            identity
            + _embed_pauli("Z", [a], n)
            + _embed_pauli("Z", [b], n)
            - _embed_pauli("ZZ", [a, b], n)
        )
    if name == "swap":
        a, b = qubits
        return 0.5 * (
            identity
            + _embed_pauli("XX", [a, b], n)
            + _embed_pauli("YY", [a, b], n)
            + _embed_pauli("ZZ", [a, b], n)
        )
    raise AssertionError(f"the dense reference does not implement gate {name!r}")


def _kraus(gate, n: int) -> list[np.ndarray]:
    """Kraus operators of one noise channel, as full `2^n x 2^n` matrices.

    Hand-written from each channel's definition, not from the engine's dual:

        depolarize(p)      (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
        dephase(p)         (1-p) rho + p Z rho Z
        pauli_channel      (1-px-py-pz) rho + px X rho X + py Y rho Y + pz Z rho Z
        depolarize2(p)     (1-p) rho + (p/15) sum over the 15 non-identity
                           two-qubit Paulis on the pair
        amplitude_damping  K0 = diag(1, sqrt(1-gamma)), K1 = sqrt(gamma)|0><1|
    """
    name = str(gate["name"])
    qubits = [int(q) for q in gate["qubits"]]
    dim = 2**n
    identity = np.eye(dim, dtype=complex)

    if name == "depolarize":
        p = float(gate["p"])
        ops = [math.sqrt(1.0 - p) * identity]
        ops += [
            math.sqrt(p / 3.0) * _embed_pauli(ch, qubits, n) for ch in ("X", "Y", "Z")
        ]
        return ops
    if name == "dephase":
        p = float(gate["p"])
        return [
            math.sqrt(1.0 - p) * identity,
            math.sqrt(p) * _embed_pauli("Z", qubits, n),
        ]
    if name == "pauli_channel":
        px, py, pz = (float(gate[k]) for k in ("px", "py", "pz"))
        ops = [math.sqrt(1.0 - px - py - pz) * identity]
        for prob, ch in ((px, "X"), (py, "Y"), (pz, "Z")):
            if prob > 0.0:
                ops.append(math.sqrt(prob) * _embed_pauli(ch, qubits, n))
        return ops
    if name == "depolarize2":
        p = float(gate["p"])
        ops = [math.sqrt(1.0 - p) * identity]
        for a, b in itertools.product("IXYZ", repeat=2):
            if a == "I" and b == "I":
                continue
            ops.append(math.sqrt(p / 15.0) * _embed_pauli(a + b, qubits, n))
        return ops
    if name == "amplitude_damping":
        gamma = float(gate["gamma"])
        k0 = np.diag([1.0 + 0j, math.sqrt(1.0 - gamma)])
        k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
        return [_embed_1q(k0, qubits[0], n), _embed_1q(k1, qubits[0], n)]
    raise AssertionError(f"the dense reference does not implement channel {name!r}")


_NOISE_NAMES = frozenset(
    {"depolarize", "dephase", "amplitude_damping", "pauli_channel", "depolarize2"}
)


def dense_final_density(spec) -> np.ndarray:
    """`rho` after running `spec` forward on `|0...0><0...0|`.

    Unitary gates act by conjugation, noise channels by `sum_k K_k rho K_k^†`.
    Gate 0 acts first, matching `Circuit` order and the engine's
    `direction="forward"`.
    """
    n = int(spec.num_qubits)
    rho = np.zeros((2**n, 2**n), dtype=complex)
    rho[0, 0] = 1.0
    for gate in spec.gates:
        if str(gate["name"]) in _NOISE_NAMES:
            rho = sum(k @ rho @ k.conj().T for k in _kraus(gate, n))
        else:
            u = _gate_unitary(gate, n)
            rho = u @ rho @ u.conj().T
    return rho


def dense_expectation(spec, observable_label: str) -> float:
    """`Tr(O rho)` for `O` the given Pauli string -- the dense answer to the same
    question `observable.propagate(..., direction="heisenberg").expectation("z+")`
    asks."""
    rho = dense_final_density(spec)
    value = np.trace(_pauli_matrix(observable_label) @ rho)
    assert abs(value.imag) < 1e-12, f"Tr(O rho) is not real: {value}"
    return float(value.real)


def dense_pauli_coefficients(rho: np.ndarray, n: int) -> dict[str, complex]:
    """`{label: Tr(P rho) / 2^n}` over all `4^n` Pauli strings.

    That normalization is the one the engine's `PauliSum` uses: a sum
    `rho = sum_P c_P P` has `c_P = Tr(P rho)/2^n`, since `Tr(P Q) = 2^n
    delta_PQ`.
    """
    dim = 2**n
    out: dict[str, complex] = {}
    for chars in itertools.product("IXYZ", repeat=n):
        label = "".join(chars)
        out[label] = complex(np.sum(_pauli_matrix(label) * rho.T) / dim)
    return out


# ==========================================================================
# The validation circuits
# ==========================================================================

VALIDATION_N = 6
FORWARD_N = 4

#: `(kind, strength)` pairs covering every noise channel the bindings expose.
#: Strengths are deliberately large (`p = 0.11`, not `1e-3`) so a wrong factor
#: cannot hide inside the tolerance.
CHANNEL_CASES = (
    ("depolarize", 0.11),
    ("dephase", 0.23),
    ("amplitude_damping", 0.17),
    ("pauli_channel", (0.05, 0.09, 0.13)),
    ("depolarize2", 0.19),
)


def _push_noise(gates: list[dict], kind, strength, qubits) -> None:
    """One noise gate object per qubit (or per pair, for `depolarize2`).

    `kind = "none"` appends nothing, which is what makes `validation_spec` do
    double duty as the unitary-only guard circuit.
    """
    if kind == "none":
        return
    if kind == "depolarize2":
        if len(qubits) == 2:
            gates.append({"name": "depolarize2", "qubits": list(qubits), "p": float(strength)})
        return
    for q in qubits:
        if kind == "pauli_channel":
            px, py, pz = strength
            gates.append(
                {"name": "pauli_channel", "qubits": [q], "px": px, "py": py, "pz": pz}
            )
        elif kind == "amplitude_damping":
            gates.append(
                {"name": "amplitude_damping", "qubits": [q], "gamma": float(strength)}
            )
        else:
            gates.append({"name": kind, "qubits": [q], "p": float(strength)})


def validation_spec(kind, strength, *, n=VALIDATION_N, layers=2, seed=7):
    """A deliberately mixed small circuit: Cliffords, non-Clifford rotations of
    all three axes, a weight-two `XY` rotation, and `kind` noise after every
    gate's support.

    Angles come from a seeded RNG so the circuit is a fixed function of `seed`
    and no angle lands on a Clifford point by accident.
    """
    rng = np.random.default_rng(seed)
    gates: list[dict] = []

    def push(gate, qubits):
        gates.append(gate)
        _push_noise(gates, kind, strength, qubits)

    for layer in range(layers):
        for q in range(n):
            axis = ("rx", "ry", "rz")[(q + layer) % 3]
            push({"name": axis, "qubits": [q], "theta": float(rng.uniform(0.3, 1.2))}, [q])
        push({"name": "h", "qubits": [0]}, [0])
        push({"name": "s", "qubits": [1]}, [1])
        for a in range(0, n - 1, 2):
            push({"name": "cnot", "qubits": [a, a + 1]}, [a, a + 1])
        for a in range(1, n - 1, 2):
            push({"name": "cz", "qubits": [a, a + 1]}, [a, a + 1])
        push({"name": "swap", "qubits": [0, n - 1]}, [0, n - 1])
        for a in range(n - 1):
            push(
                {
                    "name": "pauli_rotation",
                    "qubits": [a, a + 1],
                    "pauli": "ZZ" if layer % 2 == 0 else "XY",
                    "theta": float(rng.uniform(0.3, 1.2)),
                },
                [a, a + 1],
            )
    return oracles.CircuitSpec(num_qubits=n, gates=tuple(gates))


def _zero_state_density_sum(n: int) -> PauliSum:
    """`|0...0><0...0|` as a `PauliSum`: `2^-n sum over subsets S of Z_S`."""
    terms = {}
    for chars in itertools.product("IZ", repeat=n):
        terms["".join(chars)] = complex(2.0**-n)
    return PauliSum.from_strings(terms, num_qubits=n)


# ==========================================================================
# 1. The dense cross-check
# ==========================================================================


@pytest.mark.parametrize("kind,strength", CHANNEL_CASES)
def test_heisenberg_expectation_matches_the_dense_density_matrix(kind, strength):
    """`<0|Phi^dagger(O)|0>` from the engine equals `Tr(O Phi(rho_0))` from an
    explicit Kraus evolution of the density matrix."""
    spec = validation_spec(kind, strength)
    q = VALIDATION_N // 2
    engine = complex(
        observables.single_z(q, VALIDATION_N)
        .propagate(spec.to_circuit(), None, direction="heisenberg")
        .expectation("z+")
    )
    label = observables.pauli_string({q: "Z"}, VALIDATION_N)
    reference = dense_expectation(spec, label)
    assert abs(engine.imag) < TOLERANCE
    assert engine.real == pytest.approx(reference, abs=TOLERANCE)

    # Two guards against a vacuous pass. A reference near zero would agree with
    # almost anything, so it must sit orders of magnitude above the tolerance;
    # and the *noiseless* circuit must give a materially different answer, or
    # the channel under test is not doing any work.
    assert abs(reference) > 1e5 * TOLERANCE, f"{kind}: reference is {reference:.3e}"
    noiseless = dense_expectation(validation_spec("none", 0.0, n=VALIDATION_N), label)
    assert abs(noiseless - reference) > 1e-2, (
        f"{kind}: noise moved <Z> by only {abs(noiseless - reference):.3e}"
    )


@pytest.mark.parametrize("kind,strength", CHANNEL_CASES)
def test_forward_direction_matches_every_dense_pauli_coefficient(kind, strength):
    """The engine's `direction="forward"` evolution of `rho_0`, coefficient by
    coefficient against the dense `rho`, over **all** `4^n` Pauli strings.

    Comparing the whole coefficient vector (not just an expectation value) is
    what makes this catch a *missing* term as well as a wrong one, and it is the
    "both directions where meaningful" half of the gate: `amplitude_damping` is
    not self-adjoint, so its forward map is a genuinely different check from the
    Heisenberg one above.
    """
    spec = validation_spec(kind, strength, n=FORWARD_N, layers=2, seed=11)
    evolved = _zero_state_density_sum(FORWARD_N).propagate(
        spec.to_circuit(), None, direction="forward"
    )
    engine = {label: coefficient for label, coefficient in oracles.pauli_terms(evolved)}
    reference = dense_pauli_coefficients(dense_final_density(spec), FORWARD_N)

    worst_label, worst = "", 0.0
    for label, want in reference.items():
        gap = abs(engine.get(label, 0.0 + 0j) - want)
        if gap > worst:
            worst_label, worst = label, gap
    assert worst < TOLERANCE, f"{kind}: worst coefficient gap {worst:.3e} at {worst_label}"
    # The trace is preserved by every channel here, so the identity coefficient
    # must stay exactly 2^-n -- an independent, hand-known invariant.
    assert engine.get("I" * FORWARD_N, 0.0 + 0j).real == pytest.approx(
        2.0**-FORWARD_N, abs=TOLERANCE
    )
    assert len(engine) > 4, f"{kind}: only {len(engine)} terms survived; nothing was tested"


def test_amplitude_damping_drives_z_to_the_plus_one_fixed_point():
    """The physics check the e42095c fix turned on: amplitude damping relaxes
    towards `|0>`, so `<Z>` *grows* towards `+1` under it -- and a qubit already
    in `|0>` stays at exactly 1, for every gamma.

    A swapped `apply`/`apply_adjoint` pair fails this: the Schrodinger map
    `I -> I + gamma Z` is non-unital and cannot be any channel's dual, and using
    it in the Heisenberg direction decays `<Z>` for `|0>` to `1 - gamma`.
    """
    n = 3
    for gamma in (0.0, 0.15, 0.5, 1.0):
        spec = oracles.CircuitSpec(
            num_qubits=n,
            gates=tuple(
                {"name": "amplitude_damping", "qubits": [q], "gamma": gamma}
                for q in range(n)
            ),
        )
        value = complex(
            observables.single_z(0, n)
            .propagate(spec.to_circuit(), None, direction="heisenberg")
            .expectation("z+")
        ).real
        assert value == pytest.approx(1.0, abs=TOLERANCE), f"gamma={gamma}"
        # Starting from |1> (the state that actually decays) gives 2*gamma - 1.
        flipped = complex(
            observables.single_z(0, n)
            .propagate(spec.to_circuit(), None, direction="heisenberg")
            .expectation("1" + "0" * (n - 1))
        ).real
        assert flipped == pytest.approx(2.0 * gamma - 1.0, abs=TOLERANCE), f"gamma={gamma}"


def test_the_dense_reference_reproduces_a_noiseless_unitary_evolution():
    """Guard on the reference itself: with no noise gate, the Kraus path must
    agree with the engine's exact (untruncated) Pauli propagation.

    Without this, a bug shared by nothing but the dense path -- a wrong CNOT
    decomposition, a transposed embedding -- would show up only as a
    hard-to-read failure in the parametrized tests above.
    """
    spec = validation_spec("none", 0.0, n=VALIDATION_N, layers=2, seed=3)
    assert spec.is_unitary
    q = VALIDATION_N // 2
    engine = complex(
        observables.single_z(q, VALIDATION_N)
        .propagate(spec.to_circuit(), None, direction="heisenberg")
        .expectation("z+")
    ).real
    reference = dense_expectation(spec, observables.pauli_string({q: "Z"}, VALIDATION_N))
    assert engine == pytest.approx(reference, abs=TOLERANCE)
    assert abs(reference) > 1e-3


def test_validation_spec_with_no_noise_emits_only_unitary_gates():
    spec = validation_spec("none", 0.0, n=3, layers=1)
    assert spec.is_unitary
    assert set(spec.gate_names) <= {
        "rx", "ry", "rz", "h", "s", "cnot", "cz", "swap", "pauli_rotation",
    }


# ==========================================================================
# 2. The showcase's noise model
# ==========================================================================


def _evolved_terms(circuit, observable, policy=None, direction="heisenberg"):
    evolved = observable.propagate(circuit, policy, direction=direction)
    return sorted(oracles.pauli_terms(evolved))


def _assert_terms_close(a, b, tol=1e-15):
    """Same Pauli strings, coefficients within `tol`.

    Coefficients are compared to a tolerance rather than for bit equality:
    equal-key summation order is unspecified and free to change (CLAUDE.md
    §Determinism policy), so a byte-exact comparison would be a tripwire, not a
    correctness statement.
    """
    assert [label for label, _ in a] == [label for label, _ in b]
    for (label, ca), (_, cb) in zip(a, b):
        assert abs(ca - cb) < tol, f"{label}: {ca} vs {cb}"


def test_p_zero_leg_matches_the_noiseless_circuit():
    """The sweep's `p = 0` leg carries `depolarize(0.0)` channels so its
    truncation schedule matches the noisy legs'. `1 - 4*0/3` is exactly `1.0`,
    the rescale is key-preserving, and re-truncating an unchanged sum drops
    nothing new -- so the leg must be the *same computation*, term for term,
    with or without those channels. The whole `p = 0` column of the sweep (and
    with it the noiseless-limit check) rests on this.
    """
    n, steps, theta_h, eps = 12, 4, 0.7, 2.0**-10
    observable = observables.single_z(n // 2, n)
    policy = truncation.coeff(eps)
    with_zero_noise = run_b2.noisy_kicked_ising(n, steps, theta_h, run_b2.depolarizing(0.0))
    noiseless = run_b2.noisy_kicked_ising(n, steps, theta_h, run_b2.NOISELESS)
    assert len(with_zero_noise) > len(noiseless)

    a = _evolved_terms(with_zero_noise, observable, policy)
    b = _evolved_terms(noiseless, observable, policy)
    assert len(a) > 50, f"only {len(a)} terms survived; the comparison is vacuous"
    _assert_terms_close(a, b)


def test_noiseless_model_reproduces_the_shared_builder():
    """`NOISELESS` must build exactly `circuits.heavy_hex_kicked_ising`'s
    circuit -- same lattice, same colored edge order, same layer order -- or the
    `p -> 0` limit would be recovering a different circuit than Benchmark C's.
    """
    n, steps, theta_h = 14, 3, 0.83
    mine = run_b2.noisy_kicked_ising(n, steps, theta_h, run_b2.NOISELESS)
    theirs = circuits.heavy_hex_kicked_ising(n, steps, theta_h, run_b2.THETA_ZZ)
    assert len(mine) == len(theirs)
    observable = observables.single_z(n // 2, n)
    _assert_terms_close(
        _evolved_terms(mine, observable), _evolved_terms(theirs, observable)
    )


@pytest.mark.parametrize("n", [8, 20, 127])
def test_channels_per_step_is_the_documented_formula(n):
    edges = circuits.heavy_hex_sublattice(n, require_connected=False)
    noiseless = run_b2.channels_per_step(n, len(edges), run_b2.NOISELESS)
    noisy = run_b2.channels_per_step(n, len(edges), run_b2.depolarizing(1e-3))
    assert noiseless == n + len(edges)
    assert noisy == 2 * n + 3 * len(edges)
    # And the formula matches what the builder actually pushes.
    built = run_b2.noisy_kicked_ising(n, 2, 0.5, run_b2.depolarizing(1e-3), edges=edges)
    assert len(built) == 2 * noisy


def test_the_marquee_channel_count_is_the_documented_686():
    assert len(circuits.heavy_hex_127_edges()) == 144
    assert run_b2.channels_per_step(127, 144, run_b2.depolarizing(1e-3)) == 686
    assert run_b2.channels_per_step(127, 144, run_b2.NOISELESS) == 271


def test_run_one_refuses_a_cutoff_below_the_safe_floor():
    with pytest.raises(ValueError, match="MIN_SAFE_COEFF"):
        run_b2.run_one(
            n=6,
            steps=1,
            theta_h=0.5,
            theta_h_label="5pi/16",
            model=run_b2.depolarizing(0.0),
            eps=1e-15,
        )


# ==========================================================================
# 3. The mechanism: why noise shrinks the tracked set
# ==========================================================================


def test_clifford_point_coefficient_is_exactly_the_hand_counted_decay():
    """At `theta_h = pi/2` (with `theta_zz = -pi/2`) the circuit is Clifford, so
    the evolved operator is a **single** Pauli string and its coefficient is a
    product of depolarizing factors with no interference. Each `depolarize(p)`
    channel multiplies by `1 - 4p/3` exactly when the string is non-identity on
    that channel's qubit, so the coefficient is `(1 - 4p/3)^hits` with `hits`
    counted from the string's own support at each noise layer.

    What is compared: the coefficient of one *full-circuit* run against a
    factor-by-factor reconstruction that walks the same gate list one gate at a
    time, applies the `1 - 4p/3` factors *by hand*, and consults the engine only
    for the Clifford support update (where the engine is already pinned against
    the dense reference above, and where each step is a single Pauli string).
    So the arithmetic under test -- which of the several hundred channels fire,
    and with what factor -- is done here, not by the engine; the engine's job in
    the reconstruction is only "where did the string move to".
    """
    n, steps, p = 9, 3, 0.06
    theta_h = math.pi / 2.0
    lam = 1.0 - 4.0 * p / 3.0
    edges = circuits.heavy_hex_sublattice(n)
    zz_order = [e for grp in circuits.heavy_hex_edge_coloring(edges) for e in grp]
    seed_q = n // 2

    # Independent bookkeeping: propagate the *support* of the single string in
    # the Heisenberg direction (gates applied in reverse) with the dense
    # reference, counting a noise hit whenever the current support contains the
    # channel's qubit. Support is read from the dense matrix, so no engine
    # internal is consulted.
    spec_gates: list[dict] = []
    for _ in range(steps):
        for q in range(n):
            spec_gates.append({"name": "rx", "qubits": [q], "theta": theta_h})
            spec_gates.append({"name": "depolarize", "qubits": [q], "p": p})
        for a, b in zz_order:
            spec_gates.append(
                {"name": "pauli_rotation", "qubits": [a, b], "pauli": "ZZ",
                 "theta": run_b2.THETA_ZZ}
            )
            for q in (a, b):
                spec_gates.append({"name": "depolarize", "qubits": [q], "p": p})

    # Track the support by evolving the seed string through the *unitary* gates
    # only, in reverse order, with `PauliSum` -- a separate, untruncated
    # propagation per prefix would be O(gates^2); instead evolve step by step.
    support = {seed_q}
    hits = 0
    dust = truncation.coeff(1e-12)
    current = observables.single_z(seed_q, n)
    for gate in reversed(spec_gates):
        if gate["name"] == "depolarize":
            if int(gate["qubits"][0]) in support:
                hits += 1
            continue
        one = oracles.CircuitSpec(num_qubits=n, gates=(gate,)).to_circuit()
        current = current.propagate(one, dust, direction="heisenberg")
        assert len(current) == 1, "a Clifford step must keep a single Pauli string"
        label = oracles.pauli_terms(current)[0][0]
        support = {i for i, ch in enumerate(label) if ch != "I"}

    circuit = run_b2.noisy_kicked_ising(n, steps, theta_h, run_b2.depolarizing(p), edges=edges)
    evolved = observables.single_z(seed_q, n).propagate(circuit, dust, direction="heisenberg")
    assert len(evolved) == 1
    label, coefficient = oracles.pauli_terms(evolved)[0]
    assert hits > 3 * steps, f"only {hits} noise hits; the test would be vacuous"
    assert abs(coefficient) == pytest.approx(lam**hits, rel=1e-12)
    assert {i for i, ch in enumerate(label) if ch != "I"} == support


def test_more_noise_keeps_a_strictly_smaller_tracked_set():
    """The headline, in miniature: at a fixed cutoff and a generic kick angle,
    every increase in `p` leaves fewer resident terms and a lower maximum
    weight."""
    n, steps, eps = 20, 6, 2.0**-10
    theta_h = run_b2.THETA_H["5pi/16"]
    observable = observables.single_z(n // 2, n)
    peaks, weights = [], []
    for p in (0.0, 5e-3, 3e-2, 1e-1):
        circuit = run_b2.noisy_kicked_ising(n, steps, theta_h, run_b2.depolarizing(p))
        evolved, stats = observable.propagate_with_stats(
            circuit, truncation.coeff(eps), direction="heisenberg"
        )
        peaks.append(stats.peak_terms)
        stats_dict = run_b2.weight_stats(evolved)
        weights.append(stats_dict["max_weight"])
    assert peaks == sorted(peaks, reverse=True), peaks
    assert peaks[0] > 10 * peaks[-1], peaks
    assert weights == sorted(weights, reverse=True), weights


def test_weight_stats_reads_a_hand_built_sum():
    n = 5
    summed = PauliSum.from_strings(
        {
            observables.pauli_string({0: "X"}, n): 1.0,
            observables.pauli_string({1: "Y", 3: "Z"}, n): 0.5,
            observables.pauli_string({0: "Z", 1: "Z", 2: "Z", 4: "X"}, n): 0.25,
        },
        num_qubits=n,
    )
    stats = run_b2.weight_stats(summed)
    assert stats["terms"] == 3
    assert stats["max_weight"] == 4
    assert stats["mean_weight"] == pytest.approx((1 + 2 + 4) / 3)
    empty = PauliSum(n)
    assert run_b2.weight_stats(empty) == {"terms": 0, "max_weight": 0, "mean_weight": 0.0}


# ==========================================================================
# 4. The noiseless limit, at a size CI can afford, and the citation
# ==========================================================================


def test_small_scale_noiseless_limit_recovers_the_exact_answer():
    """The `p -> 0` check of the driver, at a size where an exact reference is a
    dense matrix rather than Benchmark C's committed file.

    Same code path as the 127-qubit version (`run_b2.noisy_kicked_ising` at
    `p = 0`, Heisenberg, `|0...0>`, a tight `min_abs_coeff`), scored against the
    hand-rolled dense evolution above; and a small nonzero `p` must move the
    answer away from it monotonically, which is what makes the limit a limit
    rather than a coincidence.
    """
    n, steps = 8, 3
    theta_h = run_b2.THETA_H["5pi/16"]
    edges = circuits.heavy_hex_sublattice(n)
    seed_q = n // 2
    observable = observables.single_z(seed_q, n)
    label = observables.pauli_string({seed_q: "Z"}, n)

    spec_gates: list[dict] = []
    for _ in range(steps):
        for q in range(n):
            spec_gates.append({"name": "rx", "qubits": [q], "theta": theta_h})
        for a, b in [
            e for grp in circuits.heavy_hex_edge_coloring(edges) for e in grp
        ]:
            spec_gates.append(
                {"name": "pauli_rotation", "qubits": [a, b], "pauli": "ZZ",
                 "theta": run_b2.THETA_ZZ}
            )
    exact = dense_expectation(
        oracles.CircuitSpec(num_qubits=n, gates=tuple(spec_gates)), label
    )

    gaps = []
    for p in (0.0, 1e-6, 1e-3, 1e-2):
        circuit = run_b2.noisy_kicked_ising(n, steps, theta_h, run_b2.depolarizing(p),
                                            edges=edges)
        value = complex(
            observable.propagate(circuit, truncation.coeff(1e-12), direction="heisenberg")
            .expectation("z+")
        ).real
        gaps.append(abs(value - exact))
    assert gaps[0] < TOLERANCE, f"p=0 must reproduce the exact value, gap {gaps[0]:.3e}"
    assert gaps == sorted(gaps), gaps
    assert gaps[-1] > 1e-3, "a p=1e-2 leg that does not move the answer proves nothing"


#: Benchmark C's claimable rows, pinned here so a change to the committed
#: `summary.json` (or to the loader) fails loudly instead of silently changing
#: what this showcase claims. Values from
#: `benchmarks/python/deep_trotter/summary.json` (commits e024d8b / 01a057c) and
#: its README §3.1.
PINNED_C_ROWS = {
    ("7pi/32", 5): (+0.655563050749, True),
    ("5pi/16", 5): (+0.238477118019, True),
    ("5pi/16", 20): (+0.016131374386, False),
}


def test_claimable_references_are_exactly_benchmark_cs_claimable_rows():
    references = run_b2.claimable_references()
    assert set(references) == set(PINNED_C_ROWS)
    assert set(run_b2.CITED_C_ROWS) == set(PINNED_C_ROWS)
    for key, (value, exact) in PINNED_C_ROWS.items():
        row = references[key]
        assert row["value"] == pytest.approx(value, abs=5e-13), key
        assert row["exact"] is exact, key
        if exact:
            assert row["uncertainty"] is None, key
            assert row["method"].startswith("light_cone_exact"), key
        else:
            assert row["method"].startswith("self_converged"), key
            assert 0.0 < row["uncertainty"] < 0.005, key
            assert row["converged"] is True, key


def test_reference_tolerance_uses_the_uncertainty_for_a_self_converged_row():
    references = run_b2.claimable_references()
    assert run_b2.reference_tolerance(references[("5pi/16", 5)]) == 0.01
    deep = references[("5pi/16", 20)]
    assert run_b2.reference_tolerance(deep) == deep["uncertainty"]
    with pytest.raises(ValueError, match="cannot be scored"):
        run_b2.reference_tolerance({"exact": False, "uncertainty": None})


def test_claimable_references_rejects_a_summary_without_claimable_rows(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"references": {}, "time_to_accuracy": []}')
    with pytest.raises(ValueError, match="no claimable row"):
        run_b2.claimable_references(path)
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="e024d8b"):
        run_b2.claimable_references(missing)
    bad = tmp_path / "bad.json"
    bad.write_text('{"references": {}}')
    with pytest.raises(ValueError, match="schema has changed"):
        run_b2.claimable_references(bad)


def test_convergence_verdict_uses_benchmark_bs_plateau_criterion():
    """The criterion is B's function object, not a re-typed copy (C does the
    same, for the same measured reason)."""
    import bench_b_theta_sweep as bench_b

    assert run_b2.bench_b._plateau_is_real is bench_b._plateau_is_real
    # A flat value on a still-growing sum is *not* convergence -- B measured
    # that failure, and the verdict must inherit it.
    flat = run_b2.convergence_verdict([0.5, 0.5, 0.5], [10, 100, 1000])
    assert flat["converged"] is False
    moving = run_b2.convergence_verdict([0.5, 0.5005, 0.50053], [10, 100, 1000])
    assert moving["converged"] is True
    assert moving["uncertainty"] == pytest.approx(3e-5, rel=1e-6)
    emptied = run_b2.convergence_verdict([0.0, 0.0, 0.0], [0, 0, 0])
    assert emptied["converged"] is False


def test_the_committed_summary_matches_the_current_grids():
    """The committed `summary.json` must describe the grid the driver would run
    today, or the README's tables and the committed SVGs are stale.

    Structure only — `p` grid, per-`p` tightest cutoff, marquee cutoff, and which
    Benchmark C rows were cited. The cut *reasons* are prose and are not compared
    here; changing one still means re-running the driver, since `summary.json`
    embeds them.
    """
    summary_path = (
        _EXAMPLES_DIR / "b2_noisy_verification" / "summary.json"
    )
    if not summary_path.exists():  # pragma: no cover - artifact removed
        pytest.skip(f"{summary_path} is not present")
    summary = json.loads(summary_path.read_text())
    assert summary["p_grid"] == list(run_b2.P_GRID)
    assert summary["coeff_grid"] == list(run_b2.COEFF_GRID)
    assert summary["marquee"]["min_abs_coeff"] == run_b2.MARQUEE_COEFF
    assert summary["marquee"]["trotter_steps"] == run_b2.MARQUEE_STEPS
    assert summary["marquee"]["theta_h_label"] == run_b2.MARQUEE_THETA_LABEL
    for p, (tightest, _reason) in run_b2.COEFF_GRID_CUTS.items():
        assert summary["coeff_grid_cuts"][str(p)]["tightest"] == run_b2.coeff_label(
            tightest
        ), p
    cited = {(r["theta_h_label"], r["trotter_steps"]) for r in summary["cited_benchmark_c_rows"]}
    assert cited == set(run_b2.CITED_C_ROWS)
    # And the legs are the p grid, each stopping at its recorded cut.
    assert [leg["p"] for leg in summary["legs"]] == list(run_b2.P_GRID)
    for leg in summary["legs"]:
        assert leg["tightest_cutoff"] == run_b2.coeff_label(
            run_b2.COEFF_GRID_CUTS[leg["p"]][0]
        ), leg["p"]


def test_the_p_grid_and_cuts_are_consistent():
    assert run_b2.P_GRID[0] == 0.0
    assert list(run_b2.P_GRID) == sorted(run_b2.P_GRID)
    assert run_b2.MARQUEE_COEFF in run_b2.COEFF_GRID
    for p in run_b2.P_GRID:
        tightest, reason = run_b2.COEFF_GRID_CUTS[p]
        assert tightest in run_b2.COEFF_GRID, p
        assert reason.strip(), p
        grid = run_b2.cutoff_grid_for(p, run_b2.COEFF_GRID, run_b2.COEFF_GRID_CUTS)
        assert grid[0] == run_b2.COEFF_GRID[0]
        assert grid[-1] == tightest
        assert min(grid) >= run_b2.MIN_SAFE_COEFF
    # Noise buys cutoff reach: the cut must loosen nowhere as p grows.
    tightest_per_p = [run_b2.COEFF_GRID_CUTS[p][0] for p in run_b2.P_GRID]
    assert tightest_per_p == sorted(tightest_per_p, reverse=True)
