"""Tests for `examples/common/circuits.py` and `observables.py` (P0a, P0b).

`examples/` is not on the pytest path by default (it isn't a package under
`python/`), so this file inserts the repo's `examples/` directory onto
`sys.path` and imports the modules as members of the top-level `common` package
-- the same pattern as `test_examples_report.py`.

What is checked here:

* the *structure* of every builder -- channel counts per Trotter step / layer,
  qubit counts, argument validation -- since the one-gate-per-channel rule
  (plan §5, D10) is what makes per-layer truncation comparable with a
  per-gate-truncating reference, and a bundled or missing gate is otherwise
  invisible;
* the Clifford points of the kicked-Ising circuit, where the exact answer is a
  single Pauli string with coefficient ±1 and no reference data is needed;
* the heavy-hex edge list against the Eagle r3 structural facts *and* against
  the generator script, so a hand edit to the checked-in file is caught;
* determinism of the Haar SU(4) sampling given a seed;
* observable term counts and weights, and that the published Kim et al.
  supports come from the provenance-tagged data file rather than a literal.

Only the generator-script test needs `qiskit-ibm-runtime`
(`pytest.importorskip`); everything else is numpy-only, so this file is CI-safe.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

from paulistrings import truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import circuits, observables  # noqa: E402


# Eagle r3, as asserted by examples/data/generate_heavy_hex.py.
EAGLE_QUBITS = 127
EAGLE_EDGES = 144


# --- heavy-hex edge list ------------------------------------------------


def test_heavy_hex_127_edge_list_is_eagle_r3():
    edges = circuits.heavy_hex_127_edges()
    assert len(edges) == EAGLE_EDGES
    assert len(set(edges)) == EAGLE_EDGES, "duplicate edges"
    assert all(a < b for a, b in edges), "edges are not stored as (lo, hi)"
    assert all(0 <= a and b < EAGLE_QUBITS for a, b in edges)

    degree: Counter = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    assert len(degree) == EAGLE_QUBITS, "some qubit has no coupling"
    assert set(degree.values()) <= {1, 2, 3}, f"degrees outside 1..3: {degree}"
    # 2*|E| = sum of degrees.
    assert sum(degree.values()) == 2 * EAGLE_EDGES
    # A heavy-hex lattice is dominated by degree-2 "bridge" qubits, with
    # degree-3 vertices only at the hexagon corners.
    assert Counter(degree.values())[3] < Counter(degree.values())[2]


def test_heavy_hex_edge_list_file_carries_provenance():
    text = circuits.HEAVY_HEX_127_PATH.read_text()
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    joined = "\n".join(header)
    assert "GENERATED FILE" in joined
    assert "generate_heavy_hex.py" in joined
    assert "FakeSherbrooke" in joined
    assert "source package" in joined


def test_heavy_hex_edge_list_matches_its_generator():
    # The checked-in file must still be what the generator produces; catches a
    # hand edit to the .edges file and a topology change on package upgrade.
    pytest.importorskip("qiskit_ibm_runtime")
    sys.path.insert(0, str(_EXAMPLES_DIR / "data"))
    try:
        import generate_heavy_hex
    finally:
        sys.path.pop(0)
    backend, _package, _version = generate_heavy_hex._load_backend()
    generated = generate_heavy_hex.extract_edges(backend)
    generate_heavy_hex.validate(generated, backend.num_qubits)
    assert generated == sorted(circuits.heavy_hex_127_edges())


def test_heavy_hex_edge_coloring_is_a_proper_three_coloring():
    edges = circuits.heavy_hex_127_edges()
    classes = circuits.heavy_hex_edge_coloring(edges)
    assert len(classes) == 3, f"expected 3 matchings, got {len(classes)}"
    assert sorted(e for group in classes for e in group) == sorted(edges)
    for group in classes:
        qubits = [q for edge in group for q in edge]
        assert len(qubits) == len(set(qubits)), "a color class is not a matching"


def test_heavy_hex_sublattices_are_induced_and_connected():
    full = set(circuits.heavy_hex_127_edges())
    for n in (2, 8, 16, 20, 32, 48, 64, 127):
        sub = circuits.heavy_hex_sublattice(n)
        assert set(sub) <= full, "sublattice edges are not device edges"
        assert all(b < n for _a, b in sub)
        # Induced: every device edge inside 0..n-1 is present.
        assert set(sub) == {e for e in full if e[1] < n}
        assert circuits._is_connected(n, sub), f"n={n} sublattice is disconnected"
    assert circuits.heavy_hex_sublattice(127) == sorted(full)


def test_disconnected_sublattice_sizes_raise_by_default():
    # Device qubits 37, 75, 113 have no lower-indexed neighbour, so a few
    # prefixes leave one isolated. The rejection is computed, not tabulated.
    with pytest.raises(ValueError, match="disconnected"):
        circuits.heavy_hex_sublattice(38)
    # ... but it is available on request, and qubit 37 is indeed the isolated one.
    sub = circuits.heavy_hex_sublattice(38, require_connected=False)
    assert 37 not in {q for edge in sub for q in edge}


def test_sublattice_size_is_bounds_checked():
    with pytest.raises(ValueError, match="1..127"):
        circuits.heavy_hex_sublattice(0)
    with pytest.raises(ValueError, match="1..127"):
        circuits.heavy_hex_sublattice(128)


def test_load_edge_list_rejects_malformed_files(tmp_path):
    bad = tmp_path / "bad.edges"
    bad.write_text("# comment\n0 1 2\n")
    with pytest.raises(ValueError, match="expected 'lo hi'"):
        circuits.load_edge_list(bad)
    bad.write_text("3 3\n")
    with pytest.raises(ValueError, match="self-loop"):
        circuits.load_edge_list(bad)
    bad.write_text("0 1\n1 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        circuits.load_edge_list(bad)


# --- kicked Ising ------------------------------------------------------


def test_kicked_ising_channel_count_is_one_gate_per_channel():
    for steps in (0, 1, 3, 5):
        circuit = circuits.heavy_hex_kicked_ising(EAGLE_QUBITS, trotter_steps=steps)
        assert circuit.num_qubits == EAGLE_QUBITS
        assert len(circuit) == steps * (EAGLE_EDGES + EAGLE_QUBITS)


def test_kicked_ising_channel_count_on_a_sublattice():
    n = 20
    n_edges = len(circuits.heavy_hex_sublattice(n))
    circuit = circuits.heavy_hex_kicked_ising(n, trotter_steps=4, theta_h=0.3)
    assert len(circuit) == 4 * (n_edges + n)


def test_kicked_ising_clifford_theta_zz_is_the_documented_angle():
    # theta_zz = -pi/2 in the exp(-i·theta·P/2) convention is exp(+i·(pi/4)·ZZ),
    # the entangler of the utility experiment.
    assert circuits.KICKED_ISING_CLIFFORD_THETA_ZZ == pytest.approx(-math.pi / 2)


def test_kicked_ising_at_theta_h_zero_leaves_a_z_observable_untouched():
    # Every generator is Z-type, so Z_62 commutes with all of them, and
    # rx(0) is the identity: the evolved observable is exactly Z_62 again.
    # No truncation, no reference data needed.
    n = EAGLE_QUBITS
    z62 = observables.single_z(62, n)
    circuit = circuits.heavy_hex_kicked_ising(n, trotter_steps=5, theta_h=0.0)
    evolved = z62.propagate(circuit=circuit, direction="heisenberg")
    assert len(evolved) == 1
    assert complex(evolved.coefficients_array()[0]) == pytest.approx(1.0 + 0j, abs=1e-12)


def test_kicked_ising_at_theta_h_half_pi_stays_a_single_pauli_string():
    # Both angles are Clifford points (exp(+i·(pi/4)·ZZ) and exp(-i·(pi/4)·X)),
    # so the evolved observable is a single Pauli string with coefficient ±1.
    # cos(pi/2) is 6.1e-17 rather than 0 in floating point, so a cutoff below
    # any physical scale is needed to drop the residues; truncation.coeff drops
    # |c| <= eps.
    n = EAGLE_QUBITS
    z62 = observables.single_z(62, n)
    circuit = circuits.heavy_hex_kicked_ising(n, trotter_steps=5, theta_h=math.pi / 2)
    evolved = z62.propagate(
        circuit=circuit, policy=truncation.coeff(1e-12), direction="heisenberg"
    )
    assert len(evolved) == 1
    assert abs(complex(evolved.coefficients_array()[0])) == pytest.approx(1.0, abs=1e-12)


def test_kicked_ising_clifford_point_on_an_anticommuting_observable():
    # X_5 does *not* commute with the ZZ generators, so this exercises the
    # fanout branch and still lands on one string at the Clifford point.
    n = 20
    x5 = observables.pauli_sum_from_support({5: "X"}, n)
    circuit = circuits.heavy_hex_kicked_ising(n, trotter_steps=3, theta_h=math.pi / 2)
    evolved = x5.propagate(
        circuit=circuit, policy=truncation.coeff(1e-12), direction="heisenberg"
    )
    assert len(evolved) == 1
    assert abs(complex(evolved.coefficients_array()[0])) == pytest.approx(1.0, abs=1e-12)


def test_kicked_ising_zz_ordering_does_not_change_the_result():
    # All ZZ generators commute, so colored and sorted emission order must give
    # the same evolved sum; only the truncation schedule differs.
    n = 20
    obs = observables.pauli_sum_from_support({5: "X", 9: "Z"}, n)
    colored = circuits.heavy_hex_kicked_ising(n, trotter_steps=2, theta_h=0.4)
    plain = circuits.heavy_hex_kicked_ising(
        n, trotter_steps=2, theta_h=0.4, color_layers=False
    )
    assert len(colored) == len(plain)
    a = obs.propagate(circuit=colored, direction="heisenberg")
    b = obs.propagate(circuit=plain, direction="heisenberg")
    _assert_sums_close(a, b)


def test_kicked_ising_final_x_layer_adds_exactly_one_layer():
    n = 20
    base = circuits.heavy_hex_kicked_ising(n, trotter_steps=3, theta_h=0.2)
    extra = circuits.heavy_hex_kicked_ising(
        n, trotter_steps=3, theta_h=0.2, final_x_layer=True
    )
    assert len(extra) == len(base) + n


def test_kicked_ising_default_order_is_the_published_one():
    assert circuits.KICKED_ISING_DEFAULT_ORDER == "x-then-zz"
    n = 20
    default = circuits.heavy_hex_kicked_ising(n, trotter_steps=2, theta_h=0.3)
    explicit = circuits.heavy_hex_kicked_ising(
        n, trotter_steps=2, theta_h=0.3, order="x-then-zz"
    )
    obs = observables.single_z(4, n)
    _assert_sums_close(
        obs.propagate(circuit=default, direction="heisenberg"),
        obs.propagate(circuit=explicit, direction="heisenberg"),
        tol=0.0,
    )


def test_kicked_ising_layer_orders_give_different_circuits():
    n = 20
    obs = observables.single_z(4, n)
    a = circuits.heavy_hex_kicked_ising(n, trotter_steps=2, theta_h=0.3, order="x-then-zz")
    b = circuits.heavy_hex_kicked_ising(n, trotter_steps=2, theta_h=0.3, order="zz-then-x")
    assert len(a) == len(b)
    assert _as_dict(obs.propagate(circuit=a, direction="heisenberg")) != _as_dict(
        obs.propagate(circuit=b, direction="heisenberg")
    )


def test_kicked_ising_rejects_bad_arguments():
    with pytest.raises(ValueError, match="trotter_steps"):
        circuits.heavy_hex_kicked_ising(8, trotter_steps=-1)
    with pytest.raises(ValueError, match="outside qubits"):
        circuits.heavy_hex_kicked_ising(4, trotter_steps=1, edges=[(0, 9)])
    with pytest.raises(ValueError, match="order must be"):
        circuits.heavy_hex_kicked_ising(8, trotter_steps=1, order="zz-x")


# --- XXZ chain ---------------------------------------------------------


def test_xxz_chain_channel_count_is_three_per_bond_per_step():
    for n, steps in ((10, 3), (25, 1), (4, 7)):
        circuit = circuits.xxz_chain_trotter(n, steps, Jz=0.7, dt=0.05)
        assert circuit.num_qubits == n
        assert len(circuit) == 3 * (n - 1) * steps
    # The channel count is independent of Jz, including the free case.
    assert len(circuits.xxz_chain_trotter(10, 2, Jz=0.0, dt=0.1)) == 3 * 9 * 2


def test_xxz_chain_bond_orders_agree_for_a_single_bond():
    # With one bond there is nothing to reorder, so the two orders must be
    # identical circuits; this pins that neither order drops a gate.
    a = circuits.xxz_chain_trotter(2, 2, Jz=0.5, dt=0.1, bond_order="even-odd")
    b = circuits.xxz_chain_trotter(2, 2, Jz=0.5, dt=0.1, bond_order="sequential")
    obs = observables.single_z(0, 2)
    _assert_sums_close(
        obs.propagate(circuit=a, direction="heisenberg"),
        obs.propagate(circuit=b, direction="heisenberg"),
    )


def test_xxz_chain_at_dt_zero_is_the_identity():
    n = 12
    circuit = circuits.xxz_chain_trotter(n, 3, Jz=0.8, dt=0.0)
    obs = observables.xxz_hamiltonian(n, Jz=0.8)
    _assert_sums_close(obs.propagate(circuit=circuit, direction="heisenberg"), obs)


def test_xxz_chain_rejects_bad_arguments():
    with pytest.raises(ValueError, match="bond_order"):
        circuits.xxz_chain_trotter(4, 1, bond_order="brickwork")
    with pytest.raises(ValueError, match="trotter_steps"):
        circuits.xxz_chain_trotter(4, -1)
    with pytest.raises(ValueError, match="n must be"):
        circuits.xxz_chain_trotter(0, 1)


# --- Haar SU(4) brickwork ---------------------------------------------


def test_haar_su4_is_special_unitary():
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(0)
    for _ in range(5):
        u = circuits.haar_su4(rng)
        assert u.shape == (4, 4)
        assert numpy.allclose(u @ u.conj().T, numpy.eye(4), atol=1e-12)
        assert abs(numpy.linalg.det(u) - 1.0) < 1e-10


def test_haar_su4_samples_differ_between_draws():
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(0)
    first, second = circuits.haar_su4(rng), circuits.haar_su4(rng)
    assert not numpy.allclose(first, second)


def test_su4_staircase_channel_count_is_brickwork():
    n, depth = 9, 5
    circuit = circuits.random_su4_staircase(n, depth, seed=7)
    assert circuit.num_qubits == n
    # Layer d covers pairs (i, i+1) with i = d mod 2, d mod 2 + 2, ...
    expected = sum(len(range(d % 2, n - 1, 2)) for d in range(depth))
    assert len(circuit) == expected


def test_su4_staircase_is_deterministic_given_the_seed():
    n = 8
    obs = observables.pauli_sum_from_support({0: "Z", 4: "X"}, n)
    a = circuits.random_su4_staircase(n, 4, seed=20260831)
    b = circuits.random_su4_staircase(n, 4, seed=20260831)
    _assert_sums_close(
        obs.propagate(circuit=a, direction="heisenberg"),
        obs.propagate(circuit=b, direction="heisenberg"),
        tol=0.0,
    )


def test_su4_staircase_differs_between_seeds():
    n = 8
    obs = observables.pauli_sum_from_support({0: "Z"}, n)
    a = obs.propagate(
        circuit=circuits.random_su4_staircase(n, 3, seed=1), direction="heisenberg"
    )
    b = obs.propagate(
        circuit=circuits.random_su4_staircase(n, 3, seed=2), direction="heisenberg"
    )
    assert _as_dict(a) != _as_dict(b)


def test_su4_staircase_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n must be"):
        circuits.random_su4_staircase(1, 2, seed=0)
    with pytest.raises(ValueError, match="depth"):
        circuits.random_su4_staircase(4, -1, seed=0)


# --- QAOA / hardware-efficient ansatz ---------------------------------


def test_qaoa_channel_count_and_qubit_inference():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    circuit = circuits.qaoa(edges, 3, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert circuit.num_qubits == 4
    assert len(circuit) == 3 * (len(edges) + 4)
    wide = circuits.qaoa(edges, 1, [0.1], [0.2], num_qubits=10)
    assert wide.num_qubits == 10
    assert len(wide) == len(edges) + 10


def test_qaoa_rejects_mismatched_parameter_counts():
    with pytest.raises(ValueError, match="gammas and betas"):
        circuits.qaoa([(0, 1)], 2, [0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="outside qubits"):
        circuits.qaoa([(0, 5)], 1, [0.1], [0.2], num_qubits=3)
    with pytest.raises(ValueError, match="self-loop"):
        circuits.qaoa([(1, 1)], 1, [0.1], [0.2])


def test_hardware_efficient_ansatz_channel_count():
    n, layers = 6, 3
    n_params = circuits.hardware_efficient_ansatz_num_params(n, layers)
    assert n_params == 2 * n * layers
    circuit = circuits.hardware_efficient_ansatz(n, layers, [0.1] * n_params)
    assert circuit.num_qubits == n
    assert len(circuit) == layers * (2 * n + (n - 1))
    cz = circuits.hardware_efficient_ansatz(
        n, layers, [0.1] * n_params, entangler="cz"
    )
    assert len(cz) == len(circuit)


def test_hardware_efficient_ansatz_rejects_bad_arguments():
    with pytest.raises(ValueError, match="expected 8 params"):
        circuits.hardware_efficient_ansatz(2, 2, [0.1] * 7)
    with pytest.raises(ValueError, match="entangler"):
        circuits.hardware_efficient_ansatz(2, 1, [0.1] * 4, entangler="iswap")


def test_hardware_efficient_ansatz_at_zero_angles_is_clifford():
    # All rotations are the identity, so only the CNOT ladder acts: a single
    # Pauli string in, a single Pauli string out.
    n = 6
    n_params = circuits.hardware_efficient_ansatz_num_params(n, 2)
    circuit = circuits.hardware_efficient_ansatz(n, 2, [0.0] * n_params)
    obs = observables.single_z(3, n)
    evolved = obs.propagate(circuit=circuit, direction="heisenberg")
    assert len(evolved) == 1


# --- observables ------------------------------------------------------


def test_pauli_string_places_characters_at_their_qubit_index():
    assert observables.pauli_string({0: "X", 3: "Z"}, 5) == "XIIZI"
    assert observables.pauli_string({}, 3) == "III"
    assert observables.pauli_string({1: "y"}, 3) == "IYI"


def test_pauli_string_validates_its_input():
    with pytest.raises(ValueError, match="outside 0..2"):
        observables.pauli_string({3: "X"}, 3)
    with pytest.raises(ValueError, match="I/X/Y/Z"):
        observables.pauli_string({0: "Q"}, 3)
    with pytest.raises(ValueError, match="num_qubits"):
        observables.pauli_string({}, 0)


def test_pauli_weight_counts_non_identity_characters():
    assert observables.pauli_weight("IIII") == 0
    assert observables.pauli_weight("XYZI") == 3


def test_single_z_is_one_term_of_weight_one():
    n = 127
    obs = observables.single_z(62, n)
    assert len(obs) == 1
    assert complex(obs.coefficients_array()[0]) == 1.0 + 0j
    # Z_62 sets bit 62 of the z plane and nothing in the x plane. n=127 is
    # W=2, so the arrays have two 64-bit words per term.
    x_words = [int(v) for v in obs.x_array()[0]]
    z_words = [int(v) for v in obs.z_array()[0]]
    assert x_words == [0, 0]
    assert z_words == [1 << 62, 0]


def test_single_z_expectation_on_the_all_zero_state():
    # |0...0> is the "z+" product state; <0|Z|0> = +1.
    assert observables.single_z(3, 8).expectation(state="z+") == pytest.approx(1.0)


def test_xxz_hamiltonian_term_count_and_weights():
    n = 10
    ham = observables.xxz_hamiltonian(n, Jz=0.5)
    assert len(ham) == 3 * (n - 1)
    free = observables.xxz_hamiltonian(n, Jz=0.0)
    assert len(free) == 2 * (n - 1)
    # Every term is a weight-2 nearest-neighbour bond, real coefficient.
    coefficients = [complex(c) for c in ham.coefficients_array()]
    assert all(abs(c.imag) < 1e-15 for c in coefficients)
    assert sorted(abs(c.real) for c in coefficients) == pytest.approx(
        sorted([1.0] * (2 * (n - 1)) + [0.5] * (n - 1))
    )


def test_xxz_hamiltonian_coupling_scales_every_term():
    n = 6
    ham = observables.xxz_hamiltonian(n, Jz=1.0, coupling=2.0)
    assert all(abs(abs(complex(c)) - 2.0) < 1e-15 for c in ham.coefficients_array())


def test_xxz_hamiltonian_needs_at_least_one_bond():
    with pytest.raises(ValueError, match="at least one bond"):
        observables.xxz_hamiltonian(1)


def test_sparse_pauli_sum_rejects_duplicates_and_mismatches():
    with pytest.raises(ValueError, match="duplicate"):
        observables.sparse_pauli_sum([{0: "X"}, {0: "X"}], [1.0, 2.0], 3)
    with pytest.raises(ValueError, match="coefficients"):
        observables.sparse_pauli_sum([{0: "X"}], [1.0, 2.0], 3)


# --- published Kim et al. supports ------------------------------------


def test_published_supports_come_only_from_the_data_file(monkeypatch, tmp_path):
    # Global rule 1: no fabricated reference values. There is no in-module
    # fallback literal, so with the data file out of reach the builders must
    # fail loudly rather than answer from a hard-coded support.
    observables._kim2023_data.cache_clear()
    monkeypatch.setattr(
        observables, "KIM2023_OBSERVABLES_PATH", tmp_path / "absent.json"
    )
    try:
        with pytest.raises(FileNotFoundError, match="provenance-tagged"):
            observables.weight_10_operator()
        with pytest.raises(FileNotFoundError, match="provenance-tagged"):
            observables.weight_17_operator()
        with pytest.raises(FileNotFoundError):
            observables.kim2023_provenance()
    finally:
        observables._kim2023_data.cache_clear()


def test_kim2023_data_file_has_a_provenance_block():
    path = observables.KIM2023_OBSERVABLES_PATH
    assert path.exists(), f"{path} is missing"
    data = json.loads(path.read_text())
    provenance = data["provenance"]
    for key in ("source", "url", "checked", "retrieved", "qubit_indexing"):
        assert provenance.get(key), f"provenance is missing {key!r}"
    assert observables.kim2023_provenance() == provenance


def test_kim2023_observable_entries_are_well_formed():
    data = json.loads(observables.KIM2023_OBSERVABLES_PATH.read_text())
    for name, entry in data["observables"].items():
        support = entry["support"]
        qubits = [int(q) for q in support]
        assert len(set(qubits)) == len(qubits), f"{name}: duplicate qubit"
        assert all(0 <= q < EAGLE_QUBITS for q in qubits), f"{name}: qubit out of range"
        assert all(op in "XYZ" for op in support.values()), f"{name}: non-XYZ operator"
        assert entry["weight"] == len(support), f"{name}: declared weight mismatch"
        assert entry.get("source_detail"), f"{name}: no per-observable source detail"


def test_declared_weights_match_the_built_observables():
    data = json.loads(observables.KIM2023_OBSERVABLES_PATH.read_text())
    for name, entry in data["observables"].items():
        obs = observables.kim2023_operator(name)
        assert len(obs) == 1, f"{name} must be a single Pauli string"
        assert obs.num_qubits == EAGLE_QUBITS
        assert complex(obs.coefficients_array()[0]) == 1.0 + 0j
        # Support read back from the symplectic bit planes -- independent of the
        # string that built it, so a Y-encoding slip would show up here.
        support = _support_of(obs)
        assert len(support) == entry["weight"] == len(entry["support"])
        assert support == {int(q): op for q, op in entry["support"].items()}


def test_unverified_observables_raise_naming_the_gap():
    data = json.loads(observables.KIM2023_OBSERVABLES_PATH.read_text())
    for name in data.get("unverified", {}):
        with pytest.raises(KeyError, match="provenance gap"):
            observables.kim2023_operator(name)


def test_unknown_observable_name_raises():
    with pytest.raises(KeyError, match="available"):
        observables.kim2023_operator("weight_42")


def test_published_supports_are_reproduced_by_the_stabilizer_relation():
    """The strongest check in this file: don't trust the transcription, redo it.

    Kim et al. state that the weight-10 and weight-17 observables are
    stabilizers of the Clifford circuit at `theta_h = pi/2`, obtained by
    evolving `Z_13` and `Z_58` for five Trotter steps (SI section VII:
    `Z(5, 13)` and `Z(5, 58)`). So the supports are *derivable*: evolve the seed
    forward through this repo's own kicked-Ising builder and the single
    resulting Pauli string must have exactly the published support and the
    published Clifford eigenvalue.

    Passing this pins four things at once: the transcribed supports in
    `kim2023_observables.json`, the heavy-hex edge list, the Trotter layer
    order (`"zz-then-x"` gives weight 6 and 15 instead of 10 and 17), and the
    correspondence between the paper's `ibm_kyiv` qubit numbering and the
    `ibm_sherbrooke` map this repo generates -- the weight-17 causal cone covers
    68 qubits, so an index permutation could not survive it.
    """
    data = json.loads(observables.KIM2023_OBSERVABLES_PATH.read_text())
    checked = 0
    for name, entry in data["observables"].items():
        origin = entry.get("stabilizer_origin")
        if origin is None:
            continue
        assert origin["theta_h"] == "pi/2", f"{name}: unexpected seed angle"
        circuit = circuits.heavy_hex_kicked_ising(
            EAGLE_QUBITS,
            trotter_steps=origin["trotter_steps"],
            theta_h=math.pi / 2,
            order=data["circuit"]["layer_order"],
            final_x_layer=origin["final_x_layer"],
        )
        seed = observables.pauli_sum_from_support(
            {origin["seed_qubit"]: origin["seed_operator"]}, EAGLE_QUBITS
        )
        # U Q U-dagger is the forward direction (see test_pauli_rotation.py).
        evolved = seed.propagate(
            circuit=circuit, policy=truncation.coeff(1e-9), direction="forward"
        )
        assert len(evolved) == 1, f"{name}: Clifford point must give one string"
        assert _support_of(evolved) == {
            int(q): op for q, op in entry["support"].items()
        }, f"{name}: evolved support does not match the published one"
        coefficient = complex(evolved.coefficients_array()[0])
        assert coefficient.real == pytest.approx(
            float(entry["clifford_eigenvalue"]), abs=1e-9
        ), f"{name}: Clifford eigenvalue mismatch"
        assert abs(coefficient.imag) < 1e-12
        checked += 1
    assert checked >= 2, "expected at least the weight-10 and weight-17 stabilizers"


def test_named_accessors_agree_with_kim2023_operator():
    pairs = [
        (observables.canonical_z_127, "weight_1_z62"),
        (observables.weight_10_operator, "weight_10"),
        (observables.weight_17_operator, "weight_17"),
        (observables.weight_17_modified_operator, "weight_17_modified"),
    ]
    for accessor, name in pairs:
        assert _support_of(accessor()) == _support_of(observables.kim2023_operator(name))
    # canonical_z_127 is the provenance-tagged spelling of single_z(62, 127).
    assert _support_of(observables.canonical_z_127()) == _support_of(
        observables.single_z(62, EAGLE_QUBITS)
    )


def test_published_operator_weights_are_as_named():
    assert len(_support_of(observables.canonical_z_127())) == 1
    assert len(_support_of(observables.weight_10_operator())) == 10
    assert len(_support_of(observables.weight_17_operator())) == 17
    assert len(_support_of(observables.weight_17_modified_operator())) == 17


def test_the_two_weight_17_operators_are_distinct():
    # Same X support, Y and Z sets swapped. Confusing them gives a silently
    # wrong observable of the same weight, so pin that they differ.
    a = observables.kim2023_operator("weight_17")
    b = observables.kim2023_operator("weight_17_modified")
    assert len(a) == len(b) == 1
    assert _support_of(a) != _support_of(b)
    xs_a = {q for q, op in _support_of(a).items() if op == "X"}
    xs_b = {q for q, op in _support_of(b).items() if op == "X"}
    assert xs_a == xs_b, "the two weight-17 operators share their X support"


def test_zz_then_x_order_does_not_reproduce_the_published_stabilizer():
    # The negative half of the order check: this is why the default is
    # "x-then-zz" and not the other way round.
    circuit = circuits.heavy_hex_kicked_ising(
        EAGLE_QUBITS, trotter_steps=5, theta_h=math.pi / 2, order="zz-then-x"
    )
    seed = observables.pauli_sum_from_support({13: "Z"}, EAGLE_QUBITS)
    evolved = seed.propagate(
        circuit=circuit, policy=truncation.coeff(1e-9), direction="forward"
    )
    assert len(evolved) == 1
    assert len(_support_of(evolved)) != 10


# --- helpers ----------------------------------------------------------


def _support_of(sum_):
    """`{qubit: 'X'|'Y'|'Z'}` of a one-term sum, read from its bit planes.

    The Hermitian convention maps `Y` to `(x=1, z=1)` with no phase factor
    (CLAUDE.md §Known gaps), so both planes set means `Y`.
    """
    assert len(sum_) == 1, f"expected a single Pauli string, got {len(sum_)}"
    x_words = [int(v) for v in sum_.x_array()[0]]
    z_words = [int(v) for v in sum_.z_array()[0]]
    support = {}
    for q in range(sum_.num_qubits):
        x_bit = (x_words[q // 64] >> (q % 64)) & 1
        z_bit = (z_words[q // 64] >> (q % 64)) & 1
        if x_bit and z_bit:
            support[q] = "Y"
        elif x_bit:
            support[q] = "X"
        elif z_bit:
            support[q] = "Z"
    return support


def _as_dict(sum_):
    xs, zs, cs = sum_.x_array(), sum_.z_array(), sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _assert_sums_close(a, b, tol=1e-12):
    da, db = _as_dict(a), _as_dict(b)
    assert set(da) == set(db), (
        f"different Pauli keys: {len(set(da) - set(db))} only in a, "
        f"{len(set(db) - set(da))} only in b"
    )
    for key, value in da.items():
        assert abs(value - db[key]) <= tol, f"{key}: {value} vs {db[key]}"
