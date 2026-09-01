"""Tests for `examples/common/oracles.py` (handoff item P0c).

`examples/` is not on the pytest path by default (it isn't a package under
`python/`), so this file inserts the repo's `examples/` directory onto
`sys.path` and imports the modules as members of the top-level `common`
package -- the same pattern as `test_examples_report.py`.

What is checked here, and why each check exists:

* **The gate-list plumbing** (`CircuitSpec`, `RecordingCircuit`,
  `record_gates`). The oracles cannot read a `paulistrings.Circuit`, so every
  one of them consumes a recorded gate list instead; if the recording drops or
  reorders a gate, every oracle silently answers a different question. The
  round-trip test therefore evolves the same observable through the recorded
  spec's `to_circuit()` and through the builder's own `Circuit` and requires
  agreement.
* **The three conversion traps** named in the module docstring: label
  endianness (qiskit reverses, stim does not), the Hermitian-Y convention (no
  phase either way), and `pauli_rotation` = `exp(-i·theta·P/2)` (reversed label
  *and* halved time in qiskit; `SQRT_P`/`SQRT_P_DAG` in stim). Each is asserted
  against a matrix or a stim named gate, not against a restatement of the claim.
* **Hand-computed expectations** on one and two qubits (`H|0>` then `Z` is 0
  and `X` is 1, and so on) -- the only numbers in this file written down by
  hand, and they are Clifford/analytic ones.
* **The independent-path gate**: `statevector_expectation` (qiskit Aer) against
  `PauliSum.propagate(..., direction="heisenberg").expectation(state)` on random
  circuits over the whole gate set. The two share no simulation code, so their
  agreement to ~1e-14 is what makes the oracle an oracle.
* **Clifford-point integers** at 127 qubits: the published Kim et al. weight-10
  / weight-17 stabilizers come back as exactly +1 / -1, and `light_cone_exact`
  reproduces those integers at both `theta_h` endpoints.
* **The reference loader refuses** files without a `source`/`method`/`accuracy`
  provenance header, since it is the enforcement point for the suite's
  no-fabricated-reference-values rule.

Per section: `pytest.importorskip("qiskit_aer")` for the statevector oracle and
`pytest.importorskip("stim")` for the Clifford oracle, so the file stays CI-safe
in the numpy-only job.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import circuits, observables, oracles  # noqa: E402


def _heisenberg_expectation(spec, observable, state="z+"):
    """The engine's own answer for the same question the oracles answer."""
    evolved = observable.propagate(spec.to_circuit(), None, direction="heisenberg")
    return complex(evolved.expectation(state))


def _random_spec(n: int, depth: int, seed: int) -> oracles.CircuitSpec:
    """A seeded circuit exercising every unitary gate the oracles translate."""
    rng = np.random.default_rng(seed)
    recorder = oracles.RecordingCircuit(n)
    for _ in range(depth):
        for q in range(n):
            kind = rng.integers(0, 6)
            if kind == 0:
                recorder.h(q)
            elif kind == 1:
                recorder.s(q)
            elif kind == 2:
                recorder.rz(float(rng.uniform(-math.pi, math.pi)), q)
            elif kind == 3:
                recorder.rx(float(rng.uniform(-math.pi, math.pi)), q)
            elif kind == 4:
                recorder.ry(float(rng.uniform(-math.pi, math.pi)), q)
            else:
                z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
                q_, r_ = np.linalg.qr(z)
                recorder.unitary_1q(q, q_ * (np.diagonal(r_) / np.abs(np.diagonal(r_))))
        for a in range(n - 1):
            kind = rng.integers(0, 5)
            if kind == 0:
                recorder.cnot(a, a + 1)
            elif kind == 1:
                recorder.cz(a, a + 1)
            elif kind == 2:
                recorder.swap(a, a + 1)
            elif kind == 3:
                pauli = "".join(rng.choice(list("XYZ"), size=2))
                recorder.pauli_rotation(
                    pauli, [a, a + 1], float(rng.uniform(-math.pi, math.pi))
                )
            else:
                recorder.unitary_2q(a, a + 1, circuits.haar_su4(rng))
    return recorder.spec


# =============================================================================
# Gate-list plumbing
# =============================================================================


def test_recording_circuit_mirrors_circuit_surface():
    recorder = oracles.RecordingCircuit(4)
    recorder.h(0)
    recorder.pauli_rotation("ZZ", [1, 2], -math.pi / 2)
    recorder.depolarize(0.1, [0, 3])
    assert recorder.num_qubits == 4
    # `depolarize` on two qubits is two channels, as it is on the real Circuit.
    assert len(recorder) == 4
    assert recorder.spec.gate_names == ("h", "pauli_rotation", "depolarize", "depolarize")
    assert recorder.spec.support(1) == (1, 2)
    assert not recorder.spec.is_unitary


def test_recording_circuit_rejects_bad_gates():
    recorder = oracles.RecordingCircuit(3)
    with pytest.raises(ValueError, match="out of range"):
        recorder.h(3)
    with pytest.raises(ValueError, match="repeats a qubit index"):
        recorder.cnot(1, 1)
    with pytest.raises(ValueError, match="characters but"):
        recorder.pauli_rotation("ZZ", [0], 0.3)
    with pytest.raises(ValueError, match="unexpected Pauli character"):
        recorder.pauli_rotation("IZ", [0, 1], 0.3)
    with pytest.raises(TypeError, match="recording gates/noise shims"):
        recorder.append(object())


def test_gate_vocabulary_matches_interop():
    """The spec's gate names must be exactly what `interop` can build.

    `CircuitSpec.to_circuit()` dispatches through
    `interop._push_task_gate`, so a name known here but not there would be a
    spec that cannot become a `Circuit`, and a name known there but not here
    would be a task-JSON circuit this module silently refuses.
    """
    from paulistrings import interop

    interop_names = set(interop._TASK_1Q) | set(interop._TASK_1Q_ROT)
    interop_names |= set(interop._TASK_2Q_NAMED) | set(interop._TASK_1Q_NOISE_P)
    interop_names |= {
        "cnot",
        "pauli_rotation",
        "amplitude_damping",
        "pauli_channel",
        "depolarize2",
        "unitary_1q",
        "unitary_2q",
    }
    assert set(oracles._GATE_SPECS) == interop_names


@pytest.mark.parametrize(
    "builder, args",
    [
        (circuits.heavy_hex_kicked_ising, (10, 2, 0.37)),
        (circuits.xxz_chain_trotter, (6, 2)),
        (circuits.random_su4_staircase, (5, 3, 11)),
        (circuits.qaoa, ([(0, 1), (1, 2), (2, 3)], 2, (0.3, 0.7), (0.5, 0.1))),
    ],
)
def test_record_gates_round_trips_through_the_engine(builder, args):
    """A recorded spec rebuilds a circuit the engine cannot distinguish.

    The comparison is an evolved expectation value rather than a channel count,
    so a recorded gate with the wrong angle, the wrong qubit order, or the wrong
    matrix convention fails here too.
    """
    spec = oracles.record_gates(builder, *args)
    direct = builder(*args)
    assert isinstance(direct, Circuit)
    assert len(spec) == len(direct)
    assert spec.num_qubits == direct.num_qubits

    n = spec.num_qubits
    observable = observables.single_z(n // 2, n)
    from_spec = observable.propagate(spec.to_circuit(), None, direction="heisenberg")
    from_direct = observable.propagate(direct, None, direction="heisenberg")
    assert from_spec.expectation("z+") == pytest.approx(
        from_direct.expectation("z+"), abs=1e-12
    )
    assert from_spec.expectation("x+") == pytest.approx(
        from_direct.expectation("x+"), abs=1e-12
    )


def test_record_gates_fails_loudly_when_it_records_nothing():
    def builder_using_an_unpatched_name():
        import paulistrings

        return paulistrings.Circuit(2)

    with pytest.raises(oracles.OracleError, match="not a RecordingCircuit"):
        oracles.record_gates(builder_using_an_unpatched_name)


def test_record_gates_restores_the_builder_globals():
    circuit_before, gates_before = circuits.Circuit, circuits.gates
    oracles.record_gates(circuits.xxz_chain_trotter, 4, 1)
    assert circuits.Circuit is circuit_before
    assert circuits.gates is gates_before
    # A leaked shim would make the *next* real build silently produce nothing.
    assert isinstance(circuits.xxz_chain_trotter(4, 1), Circuit)


def test_as_circuit_spec_rejects_an_opaque_circuit():
    with pytest.raises(TypeError, match="record_gates"):
        oracles.as_circuit_spec(Circuit(3))


def test_circuit_spec_round_trips_through_task_json():
    spec = _random_spec(3, 1, seed=5)
    document = spec.to_circuit_json()
    # Task JSON must be JSON-serializable: matrices become [re, im] pairs.
    reloaded = oracles.as_circuit_spec(
        json.loads(json.dumps(document)), num_qubits=spec.num_qubits
    )
    assert reloaded.gate_names == spec.gate_names
    observable = observables.single_z(1, 3)
    assert _heisenberg_expectation(reloaded, observable) == pytest.approx(
        _heisenberg_expectation(spec, observable), abs=1e-12
    )


# =============================================================================
# Observable decoding and the label conventions
# =============================================================================


def test_pauli_terms_decodes_a_pauli_sum():
    terms = {"XIZ": 2.0, "IYI": -1.5j}
    decoded = dict(oracles.pauli_terms(PauliSum.from_strings(terms, num_qubits=3)))
    assert decoded == {"XIZ": 2 + 0j, "IYI": -1.5j}


def test_pauli_terms_decodes_across_a_word_boundary():
    """Qubit 64 lives in the second `u64` word; a wrong word index is silent."""
    label = "I" * 64 + "Y" + "I" * 35
    (decoded,) = oracles.pauli_terms(PauliSum.from_strings({label: 1.0}, num_qubits=100))
    assert decoded == (label, 1 + 0j)


def test_pauli_terms_is_idempotent():
    terms = oracles.pauli_terms({"XZ": 1.0})
    assert oracles.pauli_terms(terms) == terms


def test_hermitian_y_needs_no_phase_in_either_direction():
    """Trap 2: both qiskit and stim spell `Y` as the Hermitian matrix."""
    qiskit_quantum_info = pytest.importorskip("qiskit.quantum_info")
    stim = pytest.importorskip("stim")
    hermitian_y = np.array([[0, -1j], [1j, 0]])
    assert np.allclose(qiskit_quantum_info.Pauli("Y").to_matrix(), hermitian_y)
    assert np.allclose(stim.PauliString("Y").to_unitary_matrix(endian="big"), hermitian_y)


def test_qiskit_labels_are_reversed_and_stim_labels_are_not():
    """Trap 1: `Z` on qubit 0 of 3 is `"ZII"` here, `"IIZ"` in qiskit, `"ZII"` in stim."""
    qiskit_quantum_info = pytest.importorskip("qiskit.quantum_info")
    stim = pytest.importorskip("stim")
    label = "ZII"
    assert oracles._to_qiskit_label(label) == "IIZ"
    # Qubit 0 is the least significant factor in qiskit, so Z_0 has diagonal
    # (+1, -1, +1, -1, ...) -- the fastest-alternating sign pattern.
    diagonal = np.diag(
        qiskit_quantum_info.SparsePauliOp(oracles._to_qiskit_label(label)).to_matrix()
    ).real
    assert list(diagonal) == [1, -1, 1, -1, 1, -1, 1, -1]
    # stim reads the label left to right by qubit index, so no reversal.
    assert stim.PauliString(label)[0] == 3  # 3 == Z
    assert stim.PauliString(label)[1] == 0


# =============================================================================
# 1. statevector_expectation
# =============================================================================


def test_statevector_hand_computed_one_qubit():
    """`H|0>` has <Z> = 0 and <X> = 1; the hand-computed base case."""
    pytest.importorskip("qiskit_aer")
    recorder = oracles.RecordingCircuit(1)
    recorder.h(0)
    spec = recorder.spec
    assert oracles.statevector_expectation(spec, "Z") == pytest.approx(0.0, abs=1e-12)
    assert oracles.statevector_expectation(spec, "X") == pytest.approx(1.0, abs=1e-12)
    assert oracles.statevector_expectation(spec, "Y") == pytest.approx(0.0, abs=1e-12)
    # S H |0> = |+i>, the +1 eigenstate of Y.
    recorder.s(0)
    assert oracles.statevector_expectation(recorder.spec, "Y") == pytest.approx(
        1.0, abs=1e-12
    )


def test_sdg_translates_to_both_oracles():
    """`sdg` reaches qiskit (`sdg`) and stim (`s_dag`) with the right sign.

    `S^dagger H |0> = |-i>`, the `-1` eigenstate of `Y`, against `S H |0> =
    |+i>` above -- so a translation that dropped the dagger flips this sign.
    """
    recorder = oracles.RecordingCircuit(1)
    recorder.h(0)
    recorder.sdg(0)
    spec = recorder.spec
    engine = _heisenberg_expectation(
        spec, observables.pauli_sum_from_support({0: "Y"}, 1)
    )
    assert engine == pytest.approx(-1.0, abs=1e-12)
    pytest.importorskip("stim")
    assert oracles.stim_clifford_exact(spec, "Y") == pytest.approx(-1.0, abs=1e-12)
    pytest.importorskip("qiskit_aer")
    assert oracles.statevector_expectation(spec, "Y") == pytest.approx(-1.0, abs=1e-12)


def test_statevector_hand_computed_two_qubit_bell():
    """`CNOT (H (x) I) |00>` is the Bell state: <ZZ> = 1, <Z_0> = <Z_1> = 0."""
    pytest.importorskip("qiskit_aer")
    recorder = oracles.RecordingCircuit(2)
    recorder.h(0)
    recorder.cnot(0, 1)
    spec = recorder.spec
    assert oracles.statevector_expectation(spec, "ZZ") == pytest.approx(1.0, abs=1e-12)
    assert oracles.statevector_expectation(spec, "XX") == pytest.approx(1.0, abs=1e-12)
    assert oracles.statevector_expectation(spec, "YY") == pytest.approx(-1.0, abs=1e-12)
    assert oracles.statevector_expectation(spec, "ZI") == pytest.approx(0.0, abs=1e-12)
    assert oracles.statevector_expectation(spec, "IZ") == pytest.approx(0.0, abs=1e-12)


def test_statevector_hand_computed_rotation_angle():
    """`rx(theta)` on `|0>` gives <Z> = cos(theta) -- pins the `exp(-i theta X/2)` scale."""
    pytest.importorskip("qiskit_aer")
    for theta in (0.0, 0.3, math.pi / 2, 1.9):
        recorder = oracles.RecordingCircuit(1)
        recorder.rx(theta, 0)
        assert oracles.statevector_expectation(recorder.spec, "Z") == pytest.approx(
            math.cos(theta), abs=1e-12
        )


def test_statevector_initial_states_are_the_product_states():
    """Every product-state spelling, against its own defining eigenvalue.

    The per-qubit alphabet is `PauliSum.expectation`'s (`0`/`1` = `Z±`, `+`/`-` =
    `X±`, `r`/`l` = `Y±`), so a wrong preparation gate here would show up as a
    sign flip against the engine in
    `test_initial_state_spellings_agree_with_the_engine`.
    """
    pytest.importorskip("qiskit_aer")
    identity = oracles.CircuitSpec(num_qubits=2, gates=())
    for state, label, expected in (
        ("z+", "ZZ", 1.0),
        ("x+", "XX", 1.0),
        ("y+", "YY", 1.0),
        ("z+", "XX", 0.0),
        ("00", "ZI", 1.0),
        ("11", "ZI", -1.0),
        ("++", "XI", 1.0),
        ("--", "XI", -1.0),
        ("rr", "YI", 1.0),
        ("ll", "YI", -1.0),
        ("01", "ZZ", -1.0),
    ):
        assert oracles.statevector_expectation(
            identity, label, state
        ) == pytest.approx(expected, abs=1e-12), (state, label)
    # A non-uniform product state, spelled as a sequence: |0>|+>.
    assert oracles.statevector_expectation(
        identity, "ZX", ["z+", "x+"]
    ) == pytest.approx(1.0, abs=1e-12)


def test_initial_state_rejections():
    identity = oracles.CircuitSpec(num_qubits=3, gates=())
    with pytest.raises(ValueError, match="are not per-qubit labels"):
        oracles._normalize_initial_state("z-", 3)
    with pytest.raises(ValueError, match="per-qubit labels for a 3-qubit"):
        oracles._normalize_initial_state("01", 3)
    with pytest.raises(ValueError, match="entries for a 3-qubit"):
        oracles._normalize_initial_state(["0", "1"], 3)
    with pytest.raises(TypeError, match="unsupported initial_state"):
        oracles._normalize_initial_state(7, 3)
    assert oracles._normalize_initial_state(None, 3) == "000"
    assert oracles._normalize_initial_state("y+", 3) == "rrr"
    assert oracles._normalize_initial_state(["z+", "-", "l"], 3) == "0-l"
    assert oracles._engine_state_argument("000") == "z+"
    assert oracles._engine_state_argument("0-l") == "0-l"
    assert identity.num_qubits == 3


@pytest.mark.parametrize("state", ["z+", "x+", "y+", "0101r", "1--rl", "rl01+"])
def test_initial_state_spellings_agree_with_the_engine(state):
    """One spelling for both engines: the oracle's prep gates vs A4 contraction.

    `PauliSum.expectation` takes the same per-qubit label string this module
    turns into preparation gates, so a disagreement here means the two
    interpretations of a character have drifted apart.
    """
    pytest.importorskip("qiskit_aer")
    n = 5
    spec = _random_spec(n, depth=2, seed=31)
    observable = PauliSum.from_strings({"XYZIZ": 1.0, "IIZZI": -0.5}, num_qubits=n)
    assert oracles.statevector_expectation(spec, observable, state) == pytest.approx(
        _heisenberg_expectation(spec, observable, state), abs=1e-12
    )


def test_pauli_rotation_matrix_is_qiskits_own_evolution():
    """Trap 3: reversed label and halved time.

    The cached matrix is compared against `exp(-i·theta·P/2)` built from a
    `SparsePauliOp` matrix directly, and the label reversal is caught by using a
    *mixed* generator, where `"XZ"` and `"ZX"` differ.
    """
    pytest.importorskip("qiskit.quantum_info")
    from qiskit.quantum_info import SparsePauliOp

    theta = 0.83
    for pauli in ("ZZ", "XZ", "YX", "XYZ"):
        matrix = oracles._pauli_rotation_matrix(pauli, theta)
        # `pauli[k]` acts on `qubits[k]`; qiskit's label is MSB-first, so the
        # generator matrix for the *same* operator uses the reversed label.
        generator = SparsePauliOp(pauli[::-1]).to_matrix()
        expected = math.cos(theta / 2) * np.eye(2 ** len(pauli)) - 1j * math.sin(
            theta / 2
        ) * generator
        assert np.allclose(matrix, expected, atol=1e-12)


def test_statevector_refuses_noise_and_oversized_systems():
    pytest.importorskip("qiskit_aer")
    recorder = oracles.RecordingCircuit(2)
    recorder.depolarize(0.1, [0])
    with pytest.raises(oracles.OracleError, match="unitary-only"):
        oracles.statevector_expectation(recorder.spec, "ZI")
    with pytest.raises(oracles.ConeTooLarge, match="max_qubits"):
        oracles.statevector_expectation(
            oracles.CircuitSpec(num_qubits=6, gates=()), "Z" + "I" * 5, max_qubits=4
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("state", ["z+", "x+", "y+"])
def test_statevector_matches_pauli_propagation(seed, state):
    """The real gate: two independent engines on a random circuit.

    qiskit Aer's dense statevector against this library's bucketed Pauli
    propagation in the Heisenberg direction. They share no simulation code, so
    agreement here is evidence about both -- and it covers every gate the
    converter translates, including the two-qubit tensor-factor order and the
    mixed-generator Pauli rotations.
    """
    pytest.importorskip("qiskit_aer")
    n = 5
    spec = _random_spec(n, depth=3, seed=seed)
    rng = np.random.default_rng(seed + 100)
    labels = ["".join(rng.choice(list("IXYZ"), size=n)) for _ in range(4)]
    coefficients = rng.normal(size=len(labels))
    observable = PauliSum.from_strings(
        {label: float(c) for label, c in zip(labels, coefficients)}, num_qubits=n
    )
    oracle = oracles.statevector_expectation(spec, observable, state)
    engine = _heisenberg_expectation(spec, observable, state)
    assert oracle == pytest.approx(engine, abs=1e-12)


# =============================================================================
# 2. stim_clifford_exact
# =============================================================================


def test_clifford_rotation_tableaux_match_stim_named_gates():
    """`exp(-i·theta·P/2)` at `theta = k·pi/2`, against stim's own gates.

    `theta = +pi/2` is `SQRT_P`, `-pi/2` (i.e. `k = 3`) is `SQRT_P_DAG`, and
    `theta = pi` is `P`. Getting the sign backwards here would flip every
    Clifford-point expectation, and the mixed/high-weight generators below have
    no named gate to check them against, so this is the anchor for those too.
    """
    stim = pytest.importorskip("stim")
    named = {
        ("X", 1): "SQRT_X",
        ("X", 3): "SQRT_X_DAG",
        ("Y", 1): "SQRT_Y",
        ("Y", 3): "SQRT_Y_DAG",
        ("Z", 1): "S",
        ("Z", 3): "S_DAG",
        ("Z", 2): "Z",
        ("X", 2): "X",
        ("XX", 1): "SQRT_XX",
        ("YY", 3): "SQRT_YY_DAG",
        ("ZZ", 1): "SQRT_ZZ",
        ("ZZ", 3): "SQRT_ZZ_DAG",
    }
    for (pauli, k), gate in named.items():
        built = oracles._clifford_rotation_tableau(stim, pauli, k)
        assert built == stim.Tableau.from_named_gate(gate), (pauli, k, gate)


def test_clifford_rotation_tableau_matches_the_unitary_for_mixed_generators():
    """The generic construction, against `exp(-i·theta·P/2)` up to global phase."""
    stim = pytest.importorskip("stim")
    from qiskit.quantum_info import SparsePauliOp

    for pauli, k in (("XZ", 1), ("ZY", 3), ("XYZ", 1), ("ZZZ", 2)):
        theta = k * math.pi / 2
        tableau = oracles._clifford_rotation_tableau(stim, pauli, k)
        # The generator's character `j` is on stim's tableau qubit `j`;
        # `endian="big"` renders tableau qubit 0 as the *most* significant
        # matrix factor, which is exactly how qiskit reads a label -- so the
        # generator matrix here uses the label unreversed. (The reversal in
        # `_pauli_rotation_matrix` is for the other endianness: qiskit's own
        # gates put `qargs[0]` in the least significant position.)
        built = tableau.to_unitary_matrix(endian="big")
        generator = SparsePauliOp(pauli).to_matrix()
        dimension = 2 ** len(pauli)
        expected = math.cos(theta / 2) * np.eye(dimension) - 1j * math.sin(
            theta / 2
        ) * generator
        # Tableaux carry no global phase, so compare the operators projectively:
        # |tr(A^dagger B)| / dim is 1 exactly when they differ by a phase. The
        # tolerance is 1e-6, not machine epsilon: stim reconstructs the unitary
        # from the tableau in single precision (`|overlap| - 1` lands around
        # 2e-8), which is why the exact anchors above go through
        # `Tableau.from_named_gate` instead.
        overlap = np.vdot(expected.ravel(), built.ravel()) / dimension
        assert abs(abs(overlap) - 1.0) < 1e-6, (pauli, k)
        assert np.allclose(built, overlap / abs(overlap) * expected, atol=1e-6)


def test_stim_known_conjugations():
    """Hand-checked stabilizer facts: `H`, `S` and `CNOT` on `|0...0>`."""
    pytest.importorskip("stim")
    recorder = oracles.RecordingCircuit(1)
    recorder.h(0)
    assert oracles.stim_clifford_exact(recorder.spec, "X") == 1
    assert oracles.stim_clifford_exact(recorder.spec, "Z") == 0
    recorder.s(0)
    # S H |0> = |+i>: <Y> = +1, and <X> = 0 because S maps X -> Y.
    assert oracles.stim_clifford_exact(recorder.spec, "Y") == 1
    assert oracles.stim_clifford_exact(recorder.spec, "X") == 0

    bell = oracles.RecordingCircuit(2)
    bell.h(0)
    bell.cnot(0, 1)
    assert oracles.stim_clifford_exact(bell.spec, "ZZ") == 1
    assert oracles.stim_clifford_exact(bell.spec, "XX") == 1
    assert oracles.stim_clifford_exact(bell.spec, "YY") == -1
    assert oracles.stim_clifford_exact(bell.spec, {"ZZ": 2.0, "XX": 3.0}) == 5


def test_stim_matches_statevector_on_a_random_clifford_circuit():
    """Two independent exact engines, on the Clifford subset of the gate set."""
    pytest.importorskip("stim")
    pytest.importorskip("qiskit_aer")
    rng = np.random.default_rng(17)
    n = 5
    recorder = oracles.RecordingCircuit(n)
    for _ in range(4):
        for q in range(n):
            kind = rng.integers(0, 4)
            if kind == 0:
                recorder.h(q)
            elif kind == 1:
                recorder.s(q)
            elif kind == 2:
                recorder.rx(float(rng.integers(0, 4)) * math.pi / 2, q)
            else:
                recorder.rz(float(rng.integers(0, 4)) * math.pi / 2, q)
        for a in range(n - 1):
            kind = rng.integers(0, 3)
            if kind == 0:
                recorder.cnot(a, a + 1)
            elif kind == 1:
                recorder.cz(a, a + 1)
            else:
                pauli = "".join(rng.choice(list("XYZ"), size=2))
                recorder.pauli_rotation(
                    pauli, [a, a + 1], float(rng.integers(1, 4)) * math.pi / 2
                )
    spec = recorder.spec
    for _ in range(6):
        label = "".join(rng.choice(list("IXYZ"), size=n))
        stim_value = oracles.stim_clifford_exact(spec, label)
        assert stim_value.imag == 0.0
        assert stim_value.real in (-1.0, 0.0, 1.0)
        assert stim_value == pytest.approx(
            oracles.statevector_expectation(spec, label), abs=1e-12
        )


def test_stim_rejects_non_clifford_and_noise():
    pytest.importorskip("stim")
    recorder = oracles.RecordingCircuit(1)
    recorder.rz(0.3, 0)
    with pytest.raises(oracles.NonCliffordGate, match="quarter-turns"):
        oracles.stim_clifford_exact(recorder.spec, "Z")

    noisy = oracles.RecordingCircuit(1)
    noisy.depolarize(0.1, [0])
    with pytest.raises(oracles.NonCliffordGate, match="noise channel"):
        oracles.stim_clifford_exact(noisy.spec, "Z")

    t_gate = oracles.RecordingCircuit(1)
    t_gate.unitary_1q(0, np.diag([1.0, np.exp(1j * math.pi / 4)]))
    with pytest.raises(oracles.NonCliffordGate, match="not a Clifford unitary"):
        oracles.stim_clifford_exact(t_gate.spec, "Z")


def test_stim_reads_a_stim_file_and_its_observable(tmp_path):
    """One `.stim` file drives both the engine's importer and this oracle."""
    stim_module = pytest.importorskip("stim")
    from paulistrings import interop

    program = "H 0\nCX 0 1\nOBSERVABLE_INCLUDE(0) Z0 Z1\n"
    path = tmp_path / "bell.stim"
    path.write_text(program)

    # Observable taken from the file.
    assert oracles.stim_clifford_exact(path) == 1
    assert oracles.stim_clifford_exact(str(path)) == 1
    assert oracles.stim_clifford_exact(stim_module.Circuit(program)) == 1
    # ... and the engine's own answer for the same file, same observable.
    circuit, observable = interop.circuit_from_stim(path)
    evolved = observable.propagate(circuit, None, direction="heisenberg")
    assert evolved.expectation("z+") == pytest.approx(1.0, abs=1e-12)


def test_stim_rejects_a_noisy_stim_file(tmp_path):
    pytest.importorskip("stim")
    path = tmp_path / "noisy.stim"
    path.write_text("H 0\nDEPOLARIZE1(0.1) 0\nOBSERVABLE_INCLUDE(0) Z0\n")
    with pytest.raises(oracles.NonCliffordGate, match="noise channel"):
        oracles.stim_clifford_exact(path)


def test_stim_needs_an_observable_when_the_file_has_none(tmp_path):
    pytest.importorskip("stim")
    path = tmp_path / "bare.stim"
    path.write_text("H 0\n")
    with pytest.raises(oracles.OracleError, match="no OBSERVABLE_INCLUDE"):
        oracles.stim_clifford_exact(path)


def test_stim_kicked_ising_clifford_point_on_a_sublattice():
    """A heavy-hex kicked Ising at `theta_h = pi/2` is Clifford, and exact.

    The rotations involved -- `exp(+i·(pi/4)·Z_iZ_j)` and `exp(-i·(pi/4)·X_q)`
    -- are Clifford but are *not* bare `H`/`S`/`CX`, which is what
    `_clifford_rotation_tableau` exists for. Cross-checked against the dense
    statevector on a lattice small enough for it.
    """
    pytest.importorskip("stim")
    pytest.importorskip("qiskit_aer")
    n = 16
    spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, n, 3, math.pi / 2)
    for q in (0, 5, n - 1):
        observable = observables.single_z(q, n)
        stim_value = oracles.stim_clifford_exact(spec, observable)
        assert stim_value.real in (-1.0, 0.0, 1.0)
        assert stim_value == pytest.approx(
            oracles.statevector_expectation(spec, observable), abs=1e-12
        )
        assert stim_value == pytest.approx(
            _heisenberg_expectation(spec, observable), abs=1e-12
        )


def test_stim_reproduces_the_published_clifford_point_integers():
    """Benchmark A's acceptance gate: exact +1 / -1 on 127 qubits.

    The Kim et al. weight-10 and weight-17 operators are stabilizers of the
    five-step `theta_h = pi/2` circuit with eigenvalues +1 and -1; the modified
    weight-17 operator is the stabilizer of the same circuit plus a final
    single-qubit rotation layer. Only Clifford-point integers may be asserted
    directly (plan §7 rule 1), and these are exactly those.
    """
    pytest.importorskip("stim")
    spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, 127, 5, math.pi / 2)
    assert oracles.stim_clifford_exact(spec, observables.weight_10_operator()) == 1
    assert oracles.stim_clifford_exact(spec, observables.weight_17_operator()) == -1

    modified = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, 127, 5, math.pi / 2, final_x_layer=True
    )
    assert (
        oracles.stim_clifford_exact(
            modified, observables.weight_17_modified_operator()
        )
        == -1
    )
    # theta_h = 0 leaves every Z diagonal untouched, so Z_62 is exactly +1.
    unkicked = oracles.record_gates(circuits.heavy_hex_kicked_ising, 127, 5, 0.0)
    assert oracles.stim_clifford_exact(unkicked, observables.single_z(62, 127)) == 1


# =============================================================================
# 3. light_cone_exact
# =============================================================================


def _heavy_hex_adjacency() -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {q: set() for q in range(127)}
    for a, b in circuits.heavy_hex_127_edges():
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _ball(adjacency, seed, radius: int) -> set[int]:
    ball = set(seed)
    for _ in range(radius):
        ball = ball | {v for u in ball for v in adjacency[u]}
    return ball


def test_light_cone_is_the_ball_around_the_observable_support():
    """One Trotter step grows the commutation-aware cone by one neighbourhood.

    With `order="x-then-zz"` a step is an `X` layer (support-preserving) then a
    `ZZ` layer, so the cone of a `Z`-type observable grows one hop per step --
    except for the *final* `ZZ` layer, which a `Z`-type operator commutes
    straight through. The `d`-step cone is therefore the radius-`(d-1)` ball,
    recomputed here from the edge list independently of the gate-list walk.
    """
    adjacency = _heavy_hex_adjacency()
    for steps in (1, 3, 5):
        spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, 127, steps, 0.4)
        cone = oracles.light_cone(spec, observables.single_z(62, 127), steps)
        assert set(cone.qubits) == _ball(adjacency, {62}, steps - 1)
        assert cone.n_steps == steps
        assert cone.source_num_qubits == 127
        assert cone.commutation_aware
        # Every kept gate lives inside the cone, so the reduced circuit closes.
        for index in cone.gate_indices:
            assert set(spec.support(index)) <= set(cone.qubits)


def test_support_only_cone_is_looser_and_still_exact():
    """The `commutation_aware=False` cone: bigger, same answer.

    Each Trotter step's `ZZ` rotations all commute, but the builder emits them
    as three disjoint-support colour classes, so the support-only walk grows
    roughly three hops per step where the commutation-aware walk grows one. Both
    reductions must give the same expectation value.
    """
    pytest.importorskip("qiskit_aer")
    n, steps = 14, 2
    spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, n, steps, 0.51)
    observable = observables.single_z(2, n)
    tight = oracles.light_cone(spec, observable, steps)
    loose = oracles.light_cone(spec, observable, steps, commutation_aware=False)
    assert set(tight.qubits) < set(loose.qubits)
    assert len(tight.gate_indices) < len(loose.gate_indices)
    assert not loose.commutation_aware
    reference = _heisenberg_expectation(spec, observable)
    for aware in (True, False):
        assert oracles.light_cone_exact(
            spec, observable, steps, commutation_aware=aware
        ) == pytest.approx(reference, abs=1e-12)


def test_light_cone_sizes_for_the_published_observables_at_five_steps():
    """The five-step cones, and their relation to the sizes the paper's SI reports.

    `examples/data/README.md` records Kim et al. SI §VII B's causal-cone sizes
    as <=31 / 37 / 68 qubits for the weight-1 / weight-10 / weight-17
    observables. Those were transcribed from the paper; here they are
    *computed*, three ways, which together check the lattice, the layer order,
    the observable supports, and the cone walk:

    * the radius-5 ball around each observable's support -- one hop per Trotter
      step, ignoring commutation -- reproduces 31 / 37 / 68 **exactly**, which
      is what the published numbers are;
    * the commutation-aware cone is one layer tighter (the trailing `ZZ` layer
      commutes through much of each observable), giving 19 / 30 / 59;
    * the support-only cone, which pays for the builder emitting each `ZZ` layer
      as three disjoint-support colour classes, is far looser: 87 / 72 / 122.

    The middle row is the one that decides `light_cone_exact`'s path at five
    steps: 19 fits the 28-qubit statevector cap, 30 and 59 do not.
    """
    adjacency = _heavy_hex_adjacency()
    spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, 127, 5, 0.4)
    observable_sets = {
        "weight_1_z62": observables.single_z(62, 127),
        "weight_10": observables.weight_10_operator(),
        "weight_17": observables.weight_17_operator(),
    }

    published_balls = {
        name: len(
            _ball(
                adjacency,
                {q for label, _ in oracles.pauli_terms(observable)
                 for q, ch in enumerate(label) if ch != "I"},
                5,
            )
        )
        for name, observable in observable_sets.items()
    }
    assert published_balls == {"weight_1_z62": 31, "weight_10": 37, "weight_17": 68}

    sizes = {
        name: oracles.light_cone(spec, observable, 5).size
        for name, observable in observable_sets.items()
    }
    assert sizes == {"weight_1_z62": 19, "weight_10": 30, "weight_17": 59}
    assert sizes["weight_1_z62"] < 68
    assert sizes["weight_1_z62"] <= oracles.DEFAULT_MAX_STATEVECTOR_QUBITS
    assert sizes["weight_10"] > oracles.DEFAULT_MAX_STATEVECTOR_QUBITS
    assert sizes["weight_17"] > oracles.DEFAULT_MAX_STATEVECTOR_QUBITS
    # Every cone is a valid over-approximation, so they must nest.
    for observable in observable_sets.values():
        tight = set(oracles.light_cone(spec, observable, 5).qubits)
        loose = set(
            oracles.light_cone(spec, observable, 5, commutation_aware=False).qubits
        )
        assert tight <= loose

    loose_sizes = {
        name: oracles.light_cone(spec, observable, 5, commutation_aware=False).size
        for name, observable in observable_sets.items()
    }
    assert loose_sizes == {"weight_1_z62": 87, "weight_10": 72, "weight_17": 122}


def test_light_cone_drops_only_causally_irrelevant_gates():
    """A gate outside the cone cannot change the answer, and is dropped."""
    pytest.importorskip("qiskit_aer")
    recorder = oracles.RecordingCircuit(6)
    recorder.h(0)
    recorder.cnot(0, 1)
    recorder.rx(0.7, 4)  # outside the cone of Z_0 Z_1
    recorder.pauli_rotation("ZZ", [4, 5], 0.3)  # ditto
    spec = recorder.spec
    observable = {"ZZIIII": 1.0}
    cone = oracles.light_cone(spec, observable)
    assert cone.qubits == (0, 1)
    assert cone.gate_indices == (0, 1)
    assert oracles.light_cone_exact(spec, observable, method="both") == pytest.approx(
        oracles.statevector_expectation(spec, observable), abs=1e-12
    )


def test_light_cone_identity_observable_is_its_trace_coefficient():
    assert oracles.light_cone_exact(
        _random_spec(3, 1, seed=9), {"III": 2.5}
    ) == pytest.approx(2.5, abs=1e-15)


@pytest.mark.parametrize("method", ["statevector", "pauli", "both"])
def test_light_cone_exact_agrees_with_the_full_problem(method):
    """The reduction is exact: the cone answer equals the un-reduced answer."""
    pytest.importorskip("qiskit_aer")
    n = 12
    spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, n, 2, 0.63)
    observable = observables.single_z(3, n)
    reference = _heisenberg_expectation(spec, observable)
    value = oracles.light_cone_exact(spec, observable, 2, method=method)
    assert value == pytest.approx(reference, abs=1e-12)


def test_light_cone_exact_endpoints_are_the_clifford_integers():
    """Plan Part A/B: the `theta_h` endpoints must reproduce A's integers."""
    pytest.importorskip("stim")
    pytest.importorskip("qiskit_aer")
    n = 16
    for theta_h in (0.0, math.pi / 2):
        spec = oracles.record_gates(circuits.heavy_hex_kicked_ising, n, 3, theta_h)
        for q in (0, 7, 12):
            observable = observables.single_z(q, n)
            expected = oracles.stim_clifford_exact(spec, observable)
            assert expected.real in (-1.0, 0.0, 1.0)
            # `method="both"` runs the dense and the untruncated-Pauli paths and
            # requires them to agree, so this pins three independent engines
            # against one integer.
            assert oracles.light_cone_exact(
                spec, observable, 3, method="both"
            ) == pytest.approx(expected, abs=1e-12)


def test_light_cone_exact_handles_noise_on_the_pauli_path_only():
    recorder = oracles.RecordingCircuit(4)
    recorder.h(0)
    recorder.cnot(0, 1)
    recorder.depolarize(0.25, [0, 1])
    spec = recorder.spec
    # <ZZ> = 1 on the Bell state; single-qubit depolarizing with probability p
    # scales each of the two Pauli factors by (1 - 4p/3).
    scale = (1.0 - 4.0 * 0.25 / 3.0) ** 2
    assert oracles.light_cone_exact(
        spec, {"ZZII": 1.0}, method="pauli"
    ) == pytest.approx(scale, abs=1e-12)
    with pytest.raises(oracles.OracleError, match="unitary-only"):
        oracles.light_cone_exact(spec, {"ZZII": 1.0}, method="statevector")


def test_unitary_only_refusal_wins_over_missing_qiskit(monkeypatch):
    """The unitary-only refusal fires before the qiskit import is attempted.

    CI's numpy-only job has no qiskit; a noisy statevector request must still
    raise the request-validity error, not SkipOracle. Reproduced here by
    forcing _import_aer to behave as if qiskit were absent.
    """

    def _no_aer():
        raise oracles.SkipOracle("qiskit deliberately unavailable in this test")

    monkeypatch.setattr(oracles, "_import_aer", _no_aer)
    recorder = oracles.RecordingCircuit(4)
    recorder.h(0)
    recorder.cnot(0, 1)
    recorder.depolarize(0.25, [0, 1])
    with pytest.raises(oracles.OracleError, match="unitary-only"):
        oracles.light_cone_exact(recorder.spec, {"ZZII": 1.0}, method="statevector")


def test_light_cone_exact_guards_and_rejections():
    recorder = oracles.RecordingCircuit(8)
    for q in range(8):
        recorder.h(q)
    for q in range(7):
        recorder.pauli_rotation("ZZ", [q, q + 1], 0.3)
    spec = recorder.spec
    # An X-type seed: the ZZ rotation on (0, 1) does not commute with it, so the
    # cone is two qubits wide (a Z-type seed would commute through and give one).
    observable = {"X" + "I" * 7: 1.0}
    assert oracles.light_cone(spec, observable).size == 2
    with pytest.raises(oracles.ConeTooLarge, match="max_statevector_qubits"):
        oracles.light_cone_exact(
            spec, observable, 1, method="statevector", max_statevector_qubits=1
        )
    with pytest.raises(ValueError, match="method must be"):
        oracles.light_cone_exact(spec, observable, method="nope")


def test_light_cone_exact_restricts_a_non_uniform_state_to_the_cone():
    """The cone's own initial state is the pattern's restriction, on both paths."""
    pytest.importorskip("qiskit_aer")
    recorder = oracles.RecordingCircuit(6)
    recorder.h(2)
    recorder.cnot(2, 3)
    recorder.rx(0.4, 5)  # outside the cone, and on a differently-prepared qubit
    spec = recorder.spec
    observable = {"IIZZII": 1.0}
    pattern = "01+-rl"
    assert oracles.light_cone(spec, observable).qubits == (2, 3)
    reference = oracles.statevector_expectation(spec, observable, pattern)
    for method in ("statevector", "pauli", "both"):
        assert oracles.light_cone_exact(
            spec, observable, initial_state=pattern, method=method
        ) == pytest.approx(reference, abs=1e-12)


def test_light_cone_exact_auto_picks_the_pauli_path_beyond_the_cap():
    """`method="auto"` must not attempt a 2**40 statevector for a 40-qubit cone.

    At seven Trotter steps the `Z_62` cone is 40 qubits, past the statevector
    cap, so `auto` has to route to the untruncated Pauli path. The run itself is
    kept cheap by taking `theta_h = 0`, where `sin(theta/2)` is *exactly* zero,
    so the sum stays a single term through all 1897 channels -- while the cone
    walk, which never reads an angle, sizes the cone as it would at any
    `theta_h`. `theta_h = pi/2` would **not** be cheap: there the dead branch
    carries `cos(pi/2) = 6.1e-17`, not `0`, so even the Clifford point fans out
    (see `light_cone_exact`'s docstring).
    """
    observable = observables.single_z(62, 127)
    generic = oracles.record_gates(circuits.heavy_hex_kicked_ising, 127, 7, 0.4)
    unkicked = oracles.record_gates(circuits.heavy_hex_kicked_ising, 127, 7, 0.0)
    cone = oracles.light_cone(generic, observable, 7)
    assert cone.size == 40 > oracles.DEFAULT_MAX_STATEVECTOR_QUBITS
    assert oracles.light_cone(unkicked, observable, 7).qubits == cone.qubits
    assert oracles.light_cone_exact(unkicked, observable, 7) == pytest.approx(
        1.0, abs=1e-12
    )


# =============================================================================
# 4. load_published_reference
# =============================================================================


def test_references_directory_ships_a_readme_and_no_data_files():
    assert (oracles.REFERENCES_DIR / "README.md").is_file()
    data_files = [
        p.name
        for p in oracles.REFERENCES_DIR.iterdir()
        if p.suffix in (".json", ".csv")
    ]
    assert data_files == [], (
        "reference data files are not checked in; if one is added, its provenance "
        "header must be recorded in the directory README in the same commit"
    )


def test_load_published_reference_reads_a_tagged_csv(tmp_path):
    (tmp_path / "ref.csv").write_text(
        "# source: https://example.invalid/dataset\n"
        "# method: exact diagonalization\n"
        "# accuracy: 1e-10 absolute\n"
        "# retrieved: 2026-08-31\n"
        "theta_h,value\n"
        "0.0,1.0\n"
        "0.5,0.75\n"
    )
    reference = oracles.load_published_reference("ref", directory=tmp_path)
    assert reference.provenance["source"] == "https://example.invalid/dataset"
    assert reference.provenance["retrieved"] == "2026-08-31"
    assert reference.fields == ("theta_h", "value")
    assert list(reference.column("value")) == [1.0, 0.75]
    with pytest.raises(KeyError, match="no column"):
        reference.column("missing")


def test_load_published_reference_reads_a_tagged_json(tmp_path):
    (tmp_path / "ref.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "source": "https://example.invalid/paper",
                    "method": "MPS, chi=1024",
                    "accuracy": "converged to 1e-3",
                },
                "data": [{"steps": 20, "value": 0.12}],
            }
        )
    )
    reference = oracles.load_published_reference("ref.json", directory=tmp_path)
    assert reference.provenance["method"] == "MPS, chi=1024"
    assert reference.rows[0]["value"] == 0.12


@pytest.mark.parametrize(
    "content, message",
    [
        ("# source: x\n# method: y\nv\n1\n", "accuracy"),
        ("# method: y\n# accuracy: z\nv\n1\n", "source"),
        ("v\n1\n", "source"),
        ("# source x\nv\n1\n", "not of the form"),
    ],
)
def test_load_published_reference_refuses_an_untagged_csv(tmp_path, content, message):
    (tmp_path / "bad.csv").write_text(content)
    with pytest.raises(oracles.OracleError, match=message):
        oracles.load_published_reference("bad", directory=tmp_path)


def test_load_published_reference_refuses_an_untagged_json(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"data": [1, 2, 3]}))
    with pytest.raises(oracles.OracleError, match='"provenance"'):
        oracles.load_published_reference("bad", directory=tmp_path)


def test_load_published_reference_names_the_available_files(tmp_path):
    (tmp_path / "other.csv").write_text("# source: a\n# method: b\n# accuracy: c\nv\n1\n")
    with pytest.raises(FileNotFoundError, match="other.csv"):
        oracles.load_published_reference("absent", directory=tmp_path)


# =============================================================================
# 5. tsim (optional)
# =============================================================================


def test_tsim_oracle_skips_cleanly_when_unavailable():
    """The optional oracle must announce itself as a skip, not a failure."""
    try:
        import tsim  # noqa: F401
    except ImportError:
        with pytest.raises(oracles.SkipOracle, match="tsim is not installed"):
            oracles.tsim_low_magic_exact(oracles.CircuitSpec(1, ()), "Z")
    else:  # pragma: no cover - tsim is not a dependency of this repo
        with pytest.raises(NotImplementedError, match="not wired up"):
            oracles.tsim_low_magic_exact(oracles.CircuitSpec(1, ()), "Z")
    assert issubclass(oracles.SkipOracle, oracles.OracleError)
