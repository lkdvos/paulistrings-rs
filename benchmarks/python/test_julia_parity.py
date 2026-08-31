"""Parity smoke gate between the Rust engine and the PauliPropagation.jl baseline.

**This gate blocks timing.** Adapted-plan global rule 2: no cross-engine
timing is reported for a run whose evolved sums diverge, so the driver has to
clear this file before any number from ``julia_baseline.py`` is trusted.

What is compared, on one shared schema-v1 task dict (so neither side can drift
into its own circuit):

1. **Per-layer term counts, exactly.** Both engines report counts in
   *application* order — jl's ``@countpaulis`` records after every gate, and
   the Rust engine's DEBUG line is ``layer {k}/{n}`` with ``k`` counting
   application steps — so for ``direction="heisenberg"`` both lists run
   backwards through the task file's gate list, and they line up index by
   index.
2. **Final expectation, to 1e-12.**

Both are checked at matched truncation on a 6-qubit circuit of h / cnot / rz /
rx — gates both engines have natively, one gate per channel (the D10
construction rule), so jl's per-gate truncation and this engine's per-channel
truncation fire at the same points.

The file also pins the two *known* semantic divergences as tests, so a future
version bump that changes either one fails loudly instead of silently shifting
term counts: the ``min_abs_coeff`` boundary (§P3 of ``benchmarks/julia/README.md``)
and the exact-zero-coefficient handling (§P9).

Run it::

    pytest benchmarks/python/test_julia_parity.py -v
    python benchmarks/python/test_julia_parity.py        # standalone report

Skips cleanly when there is no ``julia`` binary or no ``paulistrings`` build.
Never imported by CI (CI runs ``pytest python/paulistrings/tests`` only).
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from julia_baseline import (  # noqa: E402  (path shim above)
    JuliaResult,
    Task,
    gate,
    importorskip_julia,
    make_task,
    run_task,
)

# Expectations must agree to this; the plan's bar.
EXPECTATION_TOL = 1e-12

# --- Rust side --------------------------------------------------------------

_LAYER_RE = re.compile(
    r"^layer (?P<k>\d+)/(?P<n>\d+) \[(?P<name>[^\]]*)\]: "
    r"(?P<before>\d+) -> (?P<after>\d+) terms"
)


class _LayerCollector(logging.Handler):
    """Collect ``(before, after)`` per layer from the engine's DEBUG records."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.layers: list[tuple[int, int]] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = _LAYER_RE.match(record.getMessage())
        if m is not None:
            self.layers.append((int(m["before"]), int(m["after"])))


def _build_rust(task: Task | Mapping[str, Any]):
    """Translate a schema-v1 task into (PauliSum, Circuit, policy, direction, state).

    ``paulistrings.interop.circuit_from_json`` (capability A5) is the shipped
    home for this translation, and application code should use it. This gate
    keeps its own minimal builder on purpose: it is the *reference* against
    which the Julia runner's independent implementation of the same frozen
    schema is checked, so it must not share code with — or depend on the
    signature of — anything still in flux. Both honour the
    one-gate-per-channel rule: one gate object, one ``Circuit`` push.
    """
    import paulistrings as ps

    payload = task.payload if isinstance(task, Task) else dict(task)
    if payload.get("version") != 1:
        raise ValueError(f"unsupported task version {payload.get('version')!r}")
    n = int(payload["n_qubits"])

    circuit = ps.Circuit(num_qubits=n)
    for g in payload["circuit"]["gates"]:
        _push_gate(circuit, g)

    observable = ps.PauliSum.from_strings(
        {
            label: (complex(v[0], v[1]) if isinstance(v, list) else complex(v))
            for label, v in payload["observable"].items()
        },
        num_qubits=n,
    )

    trunc = payload.get("truncation", {})
    policy = None
    if "max_weight" in trunc:
        policy = ps.truncation.weight(int(trunc["max_weight"]))
    if "min_abs_coeff" in trunc:
        c = ps.truncation.coeff(float(trunc["min_abs_coeff"]))
        policy = c if policy is None else (policy & c)

    run = payload["run"]
    return observable, circuit, policy, run["direction"], run.get("state")


def _push_gate(circuit, g: Mapping[str, Any]) -> None:
    """One schema-v1 gate object -> one channel on ``circuit``."""
    name = g["name"]
    qs = list(g["qubits"])
    if name in ("h", "s", "x", "y", "z"):
        getattr(circuit, name)(qs[0])
    elif name == "cnot":
        circuit.cnot(qs[0], qs[1])
    elif name == "cz":
        circuit.cz(qs[0], qs[1])
    elif name == "swap":
        circuit.swap(qs[0], qs[1])
    elif name in ("rz", "rx", "ry"):
        getattr(circuit, name)(float(g["theta"]), qs[0])
    elif name == "pauli_rotation":
        circuit.pauli_rotation(g["pauli"], qs, float(g["theta"]))
    elif name == "depolarize":
        circuit.depolarize(float(g["p"]), [qs[0]])
    elif name == "dephase":
        circuit.dephase(float(g["p"]), [qs[0]])
    elif name == "amplitude_damping":
        circuit.amplitude_damping(float(g["gamma"]), [qs[0]])
    elif name == "pauli_channel":
        circuit.pauli_channel(float(g["px"]), float(g["py"]), float(g["pz"]), [qs[0]])
    elif name == "depolarize2":
        circuit.depolarize2(float(g["p"]), [(qs[0], qs[1])])
    elif name in ("unitary_1q", "unitary_2q"):
        import numpy as np

        mat = np.array(
            [
                [complex(e[0], e[1]) if isinstance(e, list) else complex(e) for e in row]
                for row in g["matrix"]
            ],
            dtype=complex,
        )
        if name == "unitary_1q":
            circuit.unitary_1q(qs[0], mat)
        else:
            circuit.unitary_2q(qs[0], qs[1], mat)
    else:
        raise ValueError(f"unknown gate name {name!r}")


def run_rust(task: Task | Mapping[str, Any]) -> dict[str, Any]:
    """Propagate ``task`` on the Rust engine, capturing per-layer term counts.

    Per-layer counts come from the DEBUG records on logger
    ``paulistrings.propagate``. ``reset_log_cache()`` is required after
    changing levels because ``pyo3-log`` caches each logger's effective level.
    The handler is removed and the level restored afterwards, since an enabled
    DEBUG filter costs a clock read per layer — never leave it on for a timed
    run (CLAUDE.md §Performance discipline).
    """
    import paulistrings as ps

    observable, circuit, policy, direction, state = _build_rust(task)

    collector = _LayerCollector()
    logger = logging.getLogger("paulistrings.propagate")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(collector)
    ps.reset_log_cache()
    try:
        evolved = observable.propagate(
            circuit=circuit, policy=policy, direction=direction
        )
    finally:
        logger.removeHandler(collector)
        logger.setLevel(old_level)
        ps.reset_log_cache()

    expectation = None
    if state is not None:
        expectation = complex(evolved.expectation(state=state))
    return {
        "input_terms": len(observable),
        "final_terms": len(evolved),
        "per_layer_terms": [after for _, after in collector.layers],
        "per_layer_in": [before for before, _ in collector.layers],
        "expectation": expectation,
        "evolved": evolved,
    }


# --- Fixtures / circuits ----------------------------------------------------

N_QUBITS = 6


def parity_gates(reps: int = 3) -> list[dict[str, Any]]:
    """A 6-qubit h / cnot / rz / rx circuit, one gate per channel.

    Angles are irrational-ish and all distinct so that no coefficient lands
    exactly on a truncation threshold and no merge cancels exactly — the two
    places where the engines are known to disagree (see the divergence tests
    below) are deliberately avoided here.
    """
    gs = [gate("h", [q]) for q in range(N_QUBITS)]
    for rep in range(reps):
        for a, b in ((0, 1), (2, 3), (4, 5)):
            gs.append(gate("cnot", [a, b]))
        for a, b in ((1, 2), (3, 4)):
            gs.append(gate("cnot", [a, b]))
        for q in range(N_QUBITS):
            gs.append(gate("rz", [q], theta=0.3137 + 0.0713 * q + 0.1109 * rep))
        for q in range(N_QUBITS):
            gs.append(gate("rx", [q], theta=0.2273 + 0.0531 * q + 0.1367 * rep))
    return gs


OBSERVABLE = {"IIZIII": 1.0, "IZIIZI": 0.5}


def parity_task(
    *,
    direction: str,
    state: str,
    min_abs_coeff: float | None = 1e-4,
    max_weight: int | None = None,
    reps: int = 3,
) -> Task:
    return make_task(
        n_qubits=N_QUBITS,
        gates=parity_gates(reps),
        observable=OBSERVABLE,
        direction=direction,
        min_abs_coeff=min_abs_coeff,
        max_weight=max_weight,
        threads=1,
        state=state,
    )


# --- Comparison -------------------------------------------------------------


def compare(task: Task) -> tuple[dict[str, Any], JuliaResult, list[str]]:
    """Run both engines on ``task``; return (rust, julia, list of mismatches)."""
    rust = run_rust(task)
    jl = run_task(task, warm_repeats=1, layer_counts=True)

    problems: list[str] = []
    if rust["final_terms"] != jl.final_terms:
        problems.append(
            f"final term count: rust={rust['final_terms']} julia={jl.final_terms}"
        )
    jl_layers = jl.per_layer_terms
    if jl_layers is None:
        problems.append("julia reported no per-layer term counts")
    else:
        rl = rust["per_layer_terms"]
        if len(rl) != len(jl_layers):
            problems.append(
                f"layer count: rust={len(rl)} julia={len(jl_layers)} "
                "(one gate object must be one channel)"
            )
        else:
            bad = [
                (i, a, b) for i, (a, b) in enumerate(zip(rl, jl_layers)) if a != b
            ]
            if bad:
                head = ", ".join(f"layer {i + 1}: {a} vs {b}" for i, a, b in bad[:8])
                problems.append(
                    f"{len(bad)}/{len(rl)} per-layer term counts differ ({head})"
                )
    re_, je = rust["expectation"], jl.expectation
    if (re_ is None) != (je is None):
        problems.append(f"expectation presence: rust={re_} julia={je}")
    elif re_ is not None and je is not None:
        if abs(re_ - je) > EXPECTATION_TOL:
            problems.append(
                f"expectation: rust={re_!r} julia={je!r} |delta|={abs(re_ - je):.3e} "
                f"> {EXPECTATION_TOL:g}"
            )
    return rust, jl, problems


# --- Tests ------------------------------------------------------------------


@pytest.fixture(scope="module")
def julia():
    return importorskip_julia()


@pytest.fixture(scope="module", autouse=True)
def _need_paulistrings():
    return pytest.importorskip("paulistrings")


@pytest.mark.parametrize(
    ("direction", "state", "min_abs_coeff", "max_weight"),
    [
        ("heisenberg", "z+", 1e-4, None),
        ("heisenberg", "x+", 1e-4, None),
        ("heisenberg", "z+", 1e-6, 4),
        ("heisenberg", "z+", None, None),
        ("forward", "z+", 1e-4, None),
    ],
)
def test_parity(julia, direction, state, min_abs_coeff, max_weight):
    task = parity_task(
        direction=direction,
        state=state,
        min_abs_coeff=min_abs_coeff,
        max_weight=max_weight,
    )
    rust, jl, problems = compare(task)
    assert not problems, (
        "PARITY FAILURE — cross-engine timing must not be reported until this is "
        "resolved:\n  " + "\n  ".join(problems)
    )
    # Guard against a vacuous pass: matching counts prove nothing about the
    # truncation semantics unless truncation actually removed terms. Compare
    # against the same circuit with no policy at all (Rust side only — this is
    # a property of the fixture, not a cross-engine claim).
    if min_abs_coeff is not None or max_weight is not None:
        loose = run_rust(
            parity_task(
                direction=direction, state=state, min_abs_coeff=None, max_weight=None
            )
        )
        assert rust["final_terms"] < loose["final_terms"], (
            f"truncation was inert ({rust['final_terms']} terms with the policy vs "
            f"{loose['final_terms']} without), so this parity case does not exercise "
            "the truncation boundary"
        )


# --- Per-gate vocabulary parity --------------------------------------------
#
# Every schema-v1 gate name gets its own single-gate task, compared
# TERM BY TERM (not just by expectation, which is blind to a Y sign that
# cancels in the contraction). The parameter mappings this exercises are the
# risky ones: `depolarize` p -> jl lambda = 4p/3, `dephase` p -> lambda = 2p,
# and the matrix Kronecker order for `unitary_2q`.

#: T = diag(1, exp(i pi/4)), entries as [re, im].
_T_GATE = [
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [math.cos(math.pi / 4), math.sin(math.pi / 4)]],
]
#: CNOT with control = first tensor factor of the matrix.
_CNOT_MATRIX = [
    [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
]

# A 3-qubit observable that puts every single-qubit Pauli somewhere on the
# gates' support, so no conjugation table entry goes unexercised.
VOCAB_OBSERVABLE = {
    "XII": 1.0,
    "YII": 0.7,
    "ZII": 0.5,
    "IXI": 0.3,
    "IZI": -0.2,
    "XZI": 0.9,
    "IIY": 0.4,
    "XYZ": 0.11,
}

VOCAB_CASES: list[tuple[str, dict[str, Any]]] = [
    ("h", gate("h", [0])),
    ("s", gate("s", [0])),
    ("x", gate("x", [0])),
    ("y", gate("y", [0])),
    ("z", gate("z", [0])),
    ("cnot", gate("cnot", [0, 1])),
    ("cnot-reversed", gate("cnot", [1, 0])),
    ("cz", gate("cz", [0, 1])),
    ("swap", gate("swap", [0, 1])),
    ("rz", gate("rz", [0], theta=0.37)),
    ("rx", gate("rx", [0], theta=0.37)),
    ("ry", gate("ry", [0], theta=0.37)),
    ("pauli_rotation-ZZ", gate("pauli_rotation", [0, 1], pauli="ZZ", theta=0.53)),
    ("pauli_rotation-XYZ", gate("pauli_rotation", [0, 1, 2], pauli="XYZ", theta=0.29)),
    ("depolarize", gate("depolarize", [0], p=0.13)),
    ("dephase", gate("dephase", [0], p=0.21)),
    # The one non-self-adjoint noise channel, so the only one whose
    # apply/apply_adjoint orientation is observable. It was excluded while the
    # core had the two swapped; see
    # test_amplitude_damping_heisenberg_is_the_unital_dual below.
    ("amplitude_damping", gate("amplitude_damping", [0], gamma=0.3)),
    ("pauli_channel", gate("pauli_channel", [0], px=0.1, py=0.05, pz=0.2)),
    ("depolarize2", gate("depolarize2", [0, 1], p=0.4)),
    ("unitary_1q-T", gate("unitary_1q", [0], matrix=_T_GATE)),
    ("unitary_2q-cnot", gate("unitary_2q", [0, 1], matrix=_CNOT_MATRIX)),
    ("unitary_2q-cnot-reversed", gate("unitary_2q", [1, 0], matrix=_CNOT_MATRIX)),
]


def _rust_terms(evolved, n_qubits: int) -> dict[str, complex]:
    """Evolved sum as ``{full-length label: coefficient}``, qubit 0 leftmost."""
    xs = evolved.x_array()
    zs = evolved.z_array()
    cs = evolved.coefficients_array()
    out: dict[str, complex] = {}
    for row in range(len(cs)):
        chars = []
        for q in range(n_qubits):
            word, bit = divmod(q, 64)
            xb = (int(xs[row][word]) >> bit) & 1
            zb = (int(zs[row][word]) >> bit) & 1
            chars.append("IXZY"[xb | (zb << 1)])
        out["".join(chars)] = complex(cs[row])
    return out


def compare_terms(
    rust_terms: Mapping[str, complex],
    jl_terms: Mapping[str, complex],
    tol: float = 1e-12,
) -> list[str]:
    problems: list[str] = []
    only_rust = sorted(set(rust_terms) - set(jl_terms))
    only_jl = sorted(set(jl_terms) - set(rust_terms))
    if only_rust:
        problems.append(f"terms only in rust: {only_rust[:8]}")
    if only_jl:
        problems.append(f"terms only in julia: {only_jl[:8]}")
    for label in sorted(set(rust_terms) & set(jl_terms)):
        d = abs(rust_terms[label] - jl_terms[label])
        if d > tol:
            problems.append(
                f"coefficient {label}: rust={rust_terms[label]!r} "
                f"julia={jl_terms[label]!r} |delta|={d:.3e}"
            )
    return problems


@pytest.mark.parametrize(
    ("label", "gate_obj"), VOCAB_CASES, ids=[c[0] for c in VOCAB_CASES]
)
def test_gate_vocabulary_parity(julia, label, gate_obj):
    task = make_task(
        n_qubits=3,
        gates=[gate_obj],
        observable=VOCAB_OBSERVABLE,
        direction="heisenberg",
        min_abs_coeff=0.0,
        threads=1,
        state="z+",
    )
    rust = run_rust(task)
    jl = run_task(task, warm_repeats=0, layer_counts=False, emit_terms=256)
    jl_terms = jl.terms
    assert jl_terms is not None, "runner did not emit terms; raise PP_EMIT_TERMS"
    problems = compare_terms(_rust_terms(rust["evolved"], 3), jl_terms)
    assert not problems, f"gate {label!r} maps differently:\n  " + "\n  ".join(problems)


def test_amplitude_damping_heisenberg_is_the_unital_dual(julia):
    """``amplitude_damping`` maps the same way as every other gate.

    ``AmplitudeDamping`` is the only built-in noise channel that is not
    self-adjoint, so it is the only one whose ``apply`` / ``apply_adjoint``
    orientation is observable at all. This test pins that orientation from both
    sides, because it was once wrong (the two were swapped, so
    ``direction="heisenberg"`` applied the Schrodinger channel; the mismatch
    against jl is what surfaced it).

    Hand derivation, Kraus ``K0 = diag(1, sqrt(1-g))``, ``K1 = sqrt(g)|0><1|``:

    * Schrodinger ``Phi(rho) = K0 rho K0' + K1 rho K1'``:
      ``I -> I + g Z``, ``Z -> (1-g) Z``, ``X, Y -> sqrt(1-g) . same``.
      Trace-preserving (``K0'K0 + K1'K1 = I``) and **not** unital.
    * Heisenberg dual ``Phi'(O) = K0' O K0 + K1' O K1``:
      ``I -> I``, ``Z -> (1-g) Z + g I``, ``X, Y -> sqrt(1-g) . same``.
      **Unital**, which is forced: the dual of a trace-preserving map fixes
      the identity.

    ``direction="heisenberg"`` must run ``Phi'`` — evolving an observable —
    and that is jl's ``heisenberg=true``. ``direction="forward"`` runs ``Phi``,
    which jl 0.8.2 has no Schrodinger picture for (see "Known gaps"), so it is
    compared against the same jl reference only to confirm it *differs* in the
    exact way the non-unital map must: no ``III`` term out of ``ZII``, and
    spurious ``Z``-carrying terms out of the identity-on-q0 observables.
    """
    gates = [gate("amplitude_damping", [0], gamma=0.3)]
    gamma = 0.3

    def task_for(direction: str) -> Task:
        return make_task(
            n_qubits=3,
            gates=gates,
            observable=VOCAB_OBSERVABLE,
            direction=direction,
            min_abs_coeff=0.0,
            threads=1,
            state="z+",
        )

    jl = run_task(
        task_for("heisenberg"), warm_repeats=0, layer_counts=False, emit_terms=256
    )
    assert jl.terms is not None

    rust_heis = _rust_terms(run_rust(task_for("heisenberg"))["evolved"], 3)
    rust_fwd = _rust_terms(run_rust(task_for("forward"))["evolved"], 3)

    heis = compare_terms(rust_heis, jl.terms)
    assert not heis, (
        "amplitude_damping disagrees with jl's heisenberg=true — check the "
        "apply/apply_adjoint orientation in channel/noise.rs:\n  "
        + "\n  ".join(heis)
    )
    # Unitality, hand-computed on the rust side too: the observable's
    # `0.5 . ZII` is the only source of an identity term, contributing
    # `0.5 . g`, and no observable term is `III` to begin with.
    assert abs(rust_heis["III"] - 0.5 * gamma) < 1e-12, rust_heis["III"]

    # The forward (Schrodinger) direction is the transpose, and must differ in
    # exactly two ways: it produces no identity term from Z, and it fans the
    # identity out to Z.
    fwd = compare_terms(rust_fwd, jl.terms)
    assert "III" not in rust_fwd, "Phi is not unital, so Z must not emit I"
    assert any("only in julia: ['III']" in p for p in fwd), fwd
    # `0.3 . IXI` picks up `0.3 . g . ZXI`, and likewise for IZI and IIY.
    assert abs(rust_fwd["ZXI"] - 0.3 * gamma) < 1e-12, rust_fwd["ZXI"]
    assert any("only in rust:" in p and "ZXI" in p for p in fwd), fwd


def test_hermitian_y_sign(julia):
    """S then rz, read out through |+>: pins the Y sign, not just |Y|.

    Hand computation (Heisenberg, gates applied in reverse written order, so
    ``s`` first then ``rz``):

        S† X S              = -Y
        U† (-Y) U, U = exp(-iθZ/2)
                            = -cos θ · Y - sin θ · X
        <+| . |+>           = -sin θ        (Y is orthogonal to |+>)

    A convention where ``Y`` carried its own phase, or where ``S`` mapped
    ``X → +Y``, would flip the sign or make the value complex.
    """
    theta = 0.4
    task = make_task(
        n_qubits=1,
        gates=[gate("rz", [0], theta=theta), gate("s", [0])],
        observable={"X": 1.0},
        direction="heisenberg",
        min_abs_coeff=1e-12,
        state="x+",
    )
    expected = -math.sin(theta)
    rust = run_rust(task)
    jl = run_task(task, warm_repeats=0, layer_counts=False)
    assert rust["expectation"] is not None
    assert jl.expectation is not None
    assert abs(rust["expectation"] - expected) < EXPECTATION_TOL, rust["expectation"]
    assert abs(jl.expectation - expected) < EXPECTATION_TOL, jl.expectation


def test_known_divergence_coefficient_boundary(julia):
    """|c| == min_abs_coeff: jl KEEPS the term, paulistrings DROPS it.

    Recorded as a finding in ``benchmarks/julia/README.md`` §P3, pinned here so
    a version bump on either side cannot change it silently. 0.25 is dyadic, so
    the coefficient lands on the threshold bit-exactly; ``z`` on a ``Z`` string
    is the identity map with sign +1, so nothing but truncation can move it.
    """
    task = make_task(
        n_qubits=1,
        gates=[gate("z", [0])],
        observable={"Z": 0.25},
        direction="heisenberg",
        min_abs_coeff=0.25,
        state="z+",
    )
    rust = run_rust(task)
    jl = run_task(task, warm_repeats=0, layer_counts=False)
    assert rust["final_terms"] == 0, "paulistrings should drop |c| <= eps"
    assert jl.final_terms == 1, "PauliPropagation.jl should keep |c| == eps"


def test_known_divergence_exact_zero(julia):
    """An exactly-cancelling merge: paulistrings drops the zero, jl keeps it.

    ``0.5·Z + (-0.5)·Z`` cannot be expressed as two task-JSON keys (the
    observable is a dict), so cancel through a gate instead: ``H`` maps
    ``X → Z`` and ``Z → X``, so ``H`` applied to ``X - Z`` gives ``Z - X``,
    never a cancellation. Use ``amplitude_damping(gamma=1)`` instead, whose
    Heisenberg action sends ``X → 0·X`` exactly.
    """
    task = make_task(
        n_qubits=1,
        gates=[gate("amplitude_damping", [0], gamma=1.0)],
        observable={"X": 1.0},
        direction="heisenberg",
        min_abs_coeff=0.0,
        state="z+",
    )
    rust = run_rust(task)
    jl = run_task(task, warm_repeats=0, layer_counts=False)
    assert rust["final_terms"] == 0, "paulistrings drops exact zeros in the merge"
    assert jl.final_terms == 1, "PauliPropagation.jl keeps an exactly-zero coefficient"


def test_forward_direction_rejects_unsupported_gates(julia):
    """jl has no Schrodinger picture for TransferMapGate / AmplitudeDamping.

    The runner rejects this up front with a message naming the gap rather than
    dying inside ``propagate``; see ``benchmarks/julia/README.md``
    ("Known gaps").
    """
    from julia_baseline import JuliaBaselineError

    task = make_task(
        n_qubits=1,
        gates=[gate("amplitude_damping", [0], gamma=0.2)],
        observable={"Z": 1.0},
        direction="forward",
        state="z+",
    )
    with pytest.raises(JuliaBaselineError, match="direction"):
        run_task(task, warm_repeats=0, layer_counts=False)


# --- Standalone report ------------------------------------------------------


def _report() -> int:
    reason = None
    try:
        import paulistrings  # noqa: F401
    except ImportError as exc:
        reason = f"paulistrings not importable: {exc}"
    if reason is None:
        from julia_baseline import skip_reason

        reason = skip_reason()
    if reason is not None:
        print(f"SKIP: {reason}")
        return 77

    cases: Sequence[tuple[str, str, float | None, int | None]] = [
        ("heisenberg", "z+", 1e-4, None),
        ("heisenberg", "x+", 1e-4, None),
        ("heisenberg", "z+", 1e-6, 4),
        ("heisenberg", "z+", None, None),
        ("forward", "z+", 1e-4, None),
    ]
    print(f"{'direction':<11} {'state':<6} {'eps':>8} {'w':>3} "
          f"{'layers':>7} {'final(rs/jl)':>14} {'|dexp|':>10}  status")
    failures = 0
    versions: dict[str, str] = {}
    for direction, state, eps, w in cases:
        task = parity_task(
            direction=direction, state=state, min_abs_coeff=eps, max_weight=w
        )
        rust, jl, problems = compare(task)
        versions = jl.versions
        d = (
            abs(rust["expectation"] - jl.expectation)
            if rust["expectation"] is not None and jl.expectation is not None
            else float("nan")
        )
        status = "OK" if not problems else "MISMATCH"
        if problems:
            failures += 1
        print(
            f"{direction:<11} {state:<6} {'-' if eps is None else format(eps, '.0e'):>8} "
            f"{'-' if w is None else w:>3} "
            f"{len(rust['per_layer_terms']):>7} "
            f"{rust['final_terms']:>6}/{jl.final_terms:<7} {d:>10.2e}  {status}"
        )
        for p in problems:
            print(f"    ! {p}")
    print()
    print(f"versions: {versions}")
    print(f"expectation tolerance: {EXPECTATION_TOL:g}")
    print("PARITY GATE: " + ("PASS" if failures == 0 else f"FAIL ({failures} case(s))"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    # The Rayon global pool spawns at `import paulistrings`, so a single-thread
    # comparison has to set the variable before the interpreter loads the
    # extension — re-exec once to guarantee it (research note A7).
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        os.environ["RAYON_NUM_THREADS"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    raise SystemExit(_report())
