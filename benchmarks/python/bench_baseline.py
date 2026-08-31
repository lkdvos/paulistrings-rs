"""Cross-library Python benchmark harness for ``paulistrings``.

Compares ``paulistrings`` against ``qiskit.quantum_info.SparsePauliOp`` and
``openfermion.QubitOperator`` on two operation groups: construction from
string terms (``construct``) and single-layer Heisenberg conjugation of a
Pauli sum by a Clifford circuit (``conjugate_clifford``). Backends that are
not installed are skipped via ``pytest.importorskip``. ``PauliStrings.jl``
is excluded — calling Julia from pytest would pull in PyJulia and isn't
worth the wiring.

Run with::

    pytest benchmarks/python --benchmark-only \\
        --benchmark-json=benchmarks/results/py.json

Layout:
* Each benchmark is parameterized over ``n_terms`` (``100``, ``1_000``,
  ``10_000``) and a fixed ``num_qubits`` so the input space scales the way
  a Hamiltonian does in practice.
* Inputs are generated once per (backend, size) via a session-scoped fixture
  using a seeded :class:`random.Random`; the bench body sees them as
  pre-built objects so we only measure the op of interest.
* Optional dependencies use ``pytest.importorskip`` — a backend missing from
  the env produces a skip, not a failure.

Comparable groups are tagged via ``@pytest.mark.benchmark(group=...)`` so the
``pytest-benchmark`` HTML report places matching ops from different libraries
side-by-side.
"""

from __future__ import annotations

import random

import pytest


# --- Test plan dimensions ---------------------------------------------------

NUM_QUBITS = 16
TERM_SIZES = [100, 1_000, 10_000]

# Single Pauli letter alphabet for random term generation. ``I`` included so
# weights are realistic — most Hamiltonian terms are not full-support.
_LETTERS = "IXYZ"


def _random_labels(n: int, num_qubits: int, seed: int) -> list[str]:
    """Generate `n` deterministic Pauli-string labels of length `num_qubits`.

    Qiskit and paulistrings both accept a left-to-right label convention with
    distinct semantics (qiskit reads MSB-first, paulistrings reads LSB-first),
    but for benchmarking throughput this difference does not affect the work
    done — each backend still processes the same number of unique terms.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    # Generating without rejection until `n` unique strings — for the sizes
    # we use (≤ 10⁴) and 16 qubits (4¹⁶ ≈ 4·10⁹ states), collisions are
    # negligible, but we still de-duplicate so all backends see exactly
    # the same number of distinct Pauli keys.
    while len(out) < n:
        s = "".join(rng.choice(_LETTERS) for _ in range(num_qubits))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _random_coeffs(n: int, seed: int) -> list[complex]:
    rng = random.Random(seed)
    return [complex(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(n)]


# --- Backend-specific input builders ---------------------------------------
# Each backend gets its own input fixture; tests pull only the backend they
# need so the others can be skipped without doing the construction work.

@pytest.fixture(scope="session")
def labels_by_size():
    return {n: _random_labels(n, NUM_QUBITS, seed=0x57E_D + n) for n in TERM_SIZES}


@pytest.fixture(scope="session")
def coeffs_by_size():
    return {n: _random_coeffs(n, seed=0xC0_FF_EE + n) for n in TERM_SIZES}


# --- 1. Construction --------------------------------------------------------
# Measures: building an N-term Pauli sum from a list of (label, coeff) pairs.
# Every backend exposes a constructor in this shape; cost is dominated by
# string parsing + hashmap insert.

@pytest.mark.parametrize("n", TERM_SIZES)
@pytest.mark.benchmark(group="construct")
def test_construct_paulistrings(benchmark, labels_by_size, coeffs_by_size, n):
    paulistrings = pytest.importorskip("paulistrings")
    labels = labels_by_size[n]
    coeffs = coeffs_by_size[n]
    terms = dict(zip(labels, coeffs))
    result = benchmark(paulistrings.PauliSum.from_strings, terms, num_qubits=NUM_QUBITS)
    assert len(result) == n


@pytest.mark.parametrize("n", TERM_SIZES)
@pytest.mark.benchmark(group="construct")
def test_construct_qiskit(benchmark, labels_by_size, coeffs_by_size, n):
    qi = pytest.importorskip("qiskit.quantum_info")
    labels = labels_by_size[n]
    coeffs = coeffs_by_size[n]
    pairs = list(zip(labels, coeffs))
    result = benchmark(qi.SparsePauliOp.from_list, pairs)
    assert len(result) == n


@pytest.mark.parametrize("n", TERM_SIZES)
@pytest.mark.benchmark(group="construct")
def test_construct_openfermion(benchmark, labels_by_size, coeffs_by_size, n):
    openfermion = pytest.importorskip("openfermion")
    labels = labels_by_size[n]
    coeffs = coeffs_by_size[n]
    # OpenFermion's QubitOperator takes a sparse term string ('X0 Z2 ...').
    # Pre-translate once, then time the constructor + summation only.
    of_specs = [
        " ".join(f"{ch}{i}" for i, ch in enumerate(s) if ch != "I")
        for s in labels
    ]
    QubitOperator = openfermion.QubitOperator

    def build():
        out = QubitOperator()
        for spec, c in zip(of_specs, coeffs):
            out += QubitOperator(spec, c)
        return out

    result = benchmark(build)
    assert len(result.terms) == n


# --- 2. Heisenberg conjugation through one Clifford layer ------------------
# Measures: ``C O C†`` for ``O`` an N-term Pauli sum and ``C`` a tiny Clifford
# circuit. Both backends do a constant-time per-term lookup plus a final
# dedupe; throughput is dominated by sort/hashing of N keys.

def _build_paulistrings_sum(n: int, labels_by_size, coeffs_by_size):
    import paulistrings as ps
    labels = labels_by_size[n]
    coeffs = coeffs_by_size[n]
    return ps.PauliSum.from_strings(dict(zip(labels, coeffs)), num_qubits=NUM_QUBITS)


def _build_qiskit_op(n: int, labels_by_size, coeffs_by_size):
    from qiskit.quantum_info import SparsePauliOp
    labels = labels_by_size[n]
    coeffs = coeffs_by_size[n]
    return SparsePauliOp.from_list(list(zip(labels, coeffs)))


@pytest.fixture(scope="session")
def paulistrings_h_cnot_circuit():
    ps = pytest.importorskip("paulistrings")
    c = ps.Circuit(num_qubits=NUM_QUBITS)
    c.h(0)
    c.cnot(0, 1)
    return c


@pytest.fixture(scope="session")
def qiskit_h_cnot_clifford():
    qiskit = pytest.importorskip("qiskit")
    qi = pytest.importorskip("qiskit.quantum_info")
    qc = qiskit.QuantumCircuit(NUM_QUBITS)
    qc.h(0)
    qc.cx(0, 1)
    return qi.Clifford(qc)


@pytest.mark.parametrize("n", TERM_SIZES)
@pytest.mark.benchmark(group="conjugate_clifford")
def test_conjugate_paulistrings(benchmark, labels_by_size, coeffs_by_size, paulistrings_h_cnot_circuit, n):
    pytest.importorskip("paulistrings")
    sum_in = _build_paulistrings_sum(n, labels_by_size, coeffs_by_size)
    circuit = paulistrings_h_cnot_circuit
    result = benchmark(sum_in.propagate, circuit=circuit, direction="heisenberg")
    # H + CNOT is unitary: term count is preserved on random inputs (no
    # cancellation expected).
    assert len(result) == n


@pytest.mark.parametrize("n", TERM_SIZES)
@pytest.mark.benchmark(group="conjugate_clifford")
def test_conjugate_qiskit(benchmark, labels_by_size, coeffs_by_size, qiskit_h_cnot_clifford, n):
    qi = pytest.importorskip("qiskit.quantum_info")
    op = _build_qiskit_op(n, labels_by_size, coeffs_by_size)
    cliff = qiskit_h_cnot_clifford

    def conjugate():
        new_paulis = op.paulis.evolve(cliff)
        # Pauli.phase is in units of `(-i)^k`; fold into coefficients.
        phases = (-1j) ** new_paulis.phase
        return qi.SparsePauliOp(new_paulis, coeffs=op.coeffs * phases).simplify()

    result = benchmark(conjugate)
    assert len(result) == n
