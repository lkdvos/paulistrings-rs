"""Showcase B5 -- CI-safe round-trip gate (adapted plan §6 Part B "B5").

`examples/b5_operator_backpropagation/run_b5.py` is the full showcase
(narrative-driven, produces the committed figures and data artifacts); this
file pins the one correctness property that matters for CI and must never
regress: back-propagating an observable through the final `k` layers of a
circuit, saving the evolved observable and the residual (front) circuit as a
schema-v1 task file, and running the residual circuit against the loaded
observable reproduces the full-circuit expectation value exactly (to
floating-point tolerance, at `policy=None`).

Deliberately tiny (`n=4`, 2 ansatz layers) so the whole file runs in a
fraction of a second and needs nothing beyond `paulistrings` + numpy --
`examples/common/{circuits,oracles,observables}.py` are numpy-only at import
time (qiskit/stim/matplotlib are only pulled in by functions this test never
calls), so this is CI-visible with no `importorskip`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from paulistrings import interop
from paulistrings import io as psio

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import circuits, observables, oracles  # noqa: E402

N_QUBITS = 4
LAYERS = 2
TAIL_K = 1
SEED = 0


def _split(spec: oracles.CircuitSpec, total_units: int, k: int):
    unit = len(spec.gates) // total_units
    split = (total_units - k) * unit
    front = oracles.CircuitSpec(num_qubits=spec.num_qubits, gates=spec.gates[:split])
    tail = oracles.CircuitSpec(num_qubits=spec.num_qubits, gates=spec.gates[split:])
    return front, tail


def _build_spec():
    n_params = circuits.hardware_efficient_ansatz_num_params(N_QUBITS, LAYERS)
    rng = np.random.default_rng(SEED)
    params = rng.uniform(0.0, 2.0 * math.pi, n_params)
    return oracles.record_gates(
        circuits.hardware_efficient_ansatz, N_QUBITS, LAYERS, params, entangler="cnot"
    )


def _observable_to_task_dict(pauli_sum):
    return {
        label: [c.real, c.imag] for label, c in oracles.pauli_terms(pauli_sum)
    }


def test_split_gate_list_recombines_to_the_full_circuit():
    spec = _build_spec()
    front, tail = _split(spec, LAYERS, TAIL_K)
    assert front.gates + tail.gates == spec.gates
    assert len(front) + len(tail) == len(spec)


def test_backpropagated_task_reproduces_full_circuit_expectation(tmp_path):
    spec = _build_spec()
    front, tail = _split(spec, LAYERS, TAIL_K)
    observable = observables.single_z(N_QUBITS // 2, N_QUBITS)

    full_circuit = spec.to_circuit()
    front_circuit = front.to_circuit()
    tail_circuit = tail.to_circuit()

    full_evolved = observable.propagate(full_circuit, None, direction="heisenberg")
    full_expectation = complex(full_evolved.expectation("z+"))

    tail_evolved = observable.propagate(tail_circuit, None, direction="heisenberg")

    # Round-trip the evolved observable through the paulistrings.io npz format
    # (A3) -- the "save the evolved observable" half of the showcase.
    npz_path = tmp_path / "evolved_observable.npz"
    psio.save(npz_path, tail_evolved)
    reloaded_tail = psio.load(npz_path)

    # Emit the schema-v1 task file: residual (front) circuit + evolved
    # observable -- the "run this shallower circuit on the QPU" artifact.
    task = {
        "version": 1,
        "n_qubits": N_QUBITS,
        "circuit": front.to_circuit_json(),
        "observable": _observable_to_task_dict(reloaded_tail),
        "run": {"direction": "heisenberg", "threads": 1, "state": "z+"},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(__import__("json").dumps(task))

    loaded = interop.load_task(task_path)
    assert loaded.direction == "heisenberg"
    assert loaded.truncation is None
    assert loaded.state == "z+"

    composed_evolved = loaded.observable.propagate(
        loaded.circuit, loaded.truncation, direction=loaded.direction
    )
    composed_expectation = complex(composed_evolved.expectation(loaded.state))

    assert abs(composed_expectation.imag) < 1e-12
    assert abs(full_expectation.imag) < 1e-12
    assert abs(composed_expectation.real - full_expectation.real) <= 1e-12


def test_backpropagated_task_reproduces_full_circuit_under_a_shared_policy():
    """The same equality holds under a nontrivial *shared* truncation policy.

    Truncation is applied after every channel regardless of which Python call
    it happens inside, so splitting the propagation into two calls (tail,
    then front) at the same policy must agree with one continuous propagate
    over the full gate list -- independent of where the split point is. This
    is the fact that makes "gap vs the exact answer = truncation error" (not
    an artifact of splitting) meaningful in the full showcase's convergence
    panel.
    """
    from paulistrings import truncation

    spec = _build_spec()
    front, tail = _split(spec, LAYERS, TAIL_K)
    observable = observables.single_z(N_QUBITS // 2, N_QUBITS)

    front_circuit = front.to_circuit()
    tail_circuit = tail.to_circuit()
    full_circuit = spec.to_circuit()

    policy = truncation.coeff(1e-3)

    composed = observable.propagate(tail_circuit, policy, direction="heisenberg").propagate(
        front_circuit, policy, direction="heisenberg"
    )
    one_shot = observable.propagate(full_circuit, policy, direction="heisenberg")

    composed_value = complex(composed.expectation("z+"))
    one_shot_value = complex(one_shot.expectation("z+"))
    assert abs(composed_value - one_shot_value) <= 1e-12


@pytest.mark.parametrize("k", [0, 1, 2])
def test_tail_depth_boundaries(k):
    """`k=0` leaves the observable untouched; `k=total_units` evolves it fully."""
    spec = _build_spec()
    front, tail = _split(spec, LAYERS, k)
    assert len(front) + len(tail) == len(spec)
    if k == 0:
        assert len(tail) == 0
    if k == LAYERS:
        assert len(front) == 0
