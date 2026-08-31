"""Channel/circuit factories.

Pin the canonical conjugation identities through the Python boundary so a
regression on the wiring is caught here rather than only in the Rust core.
"""

import math

import pytest

from paulistrings import Circuit, PauliSum, gates, noise


TOL = 1e-12


def coeffs_close(actual, expected, tol=TOL):
    if len(actual) != len(expected):
        return False
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


def dominant_coeff(coefficients, tol=1e-10):
    """Return the largest-magnitude coefficient and assert all others are < tol.

    Pauli rotations at angles like ``pi`` leave ``sin(pi)`` residues on the
    fan-out term — not exactly zero, so the merge keeps them when there is no
    coefficient threshold policy. The hand-computed Pauli identities are still
    correct up to the dominant term, which is what we check.
    """
    assert coefficients, "expected at least one coefficient"
    dom = max(coefficients, key=lambda c: abs(c))
    for c in coefficients:
        if c is dom:
            continue
        assert abs(c) < tol, f"unexpected non-residue coefficient: {c}"
    return dom


def test_circuit_h_conjugates_z_to_x():
    # Heisenberg picture: H Z H = X.
    s = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.h(0)
    out = s.propagate(c)
    assert out.num_qubits == 1
    # X on qubit 0 has key (x=1, z=0).
    assert coeffs_close(out.coefficients(), [1 + 0j])
    # The X-key Pauli round-trips back to Z under H twice.
    c2 = Circuit(1)
    c2.h(0)
    c2.h(0)
    out2 = s.propagate(c2)
    assert coeffs_close(out2.coefficients(), [1 + 0j])


def test_gates_factories_via_append():
    # Same H Z H = X test, but going through gates.h(...) + Circuit.append.
    s = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.append(gates.h(0))
    assert len(c) == 1
    out = s.propagate(c)
    assert coeffs_close(out.coefficients(), [1 + 0j])


def test_cnot_propagates_z_i_to_z_z():
    # CNOT (control=0, target=1) maps I⊗Z → Z⊗Z under conjugation.
    # In our string convention "IZ" = I on qubit 0, Z on qubit 1 = (x=0, z=2).
    s = PauliSum.from_strings({"IZ": 1.0}, num_qubits=2)
    c = Circuit(2)
    c.cnot(0, 1)
    out = s.propagate(c)
    assert coeffs_close(out.coefficients(), [1 + 0j])
    # The output should be the single key Z⊗Z = "ZZ" = (x=0, z=3); just
    # assert length and coefficient here — see test_numpy.py for the
    # x_array()/z_array() bit-layout checks.
    assert len(out) == 1


def test_rz_pi_flips_x_to_minus_x():
    # exp(-i·π·Z/2) X exp(+i·π·Z/2) = -X (sign in coefficient).
    s = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.rz(math.pi, 0)
    out = s.propagate(c)
    dom = dominant_coeff(out.coefficients())
    assert abs(dom - (-1 + 0j)) < TOL


def test_rx_and_ry_factories():
    # rx(π) sends Z → -Z (with the second term having sin(π)≈0 residue).
    s = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.append(gates.rx(math.pi, 0))
    out = s.propagate(c)
    dom = dominant_coeff(out.coefficients())
    assert abs(dom - (-1 + 0j)) < TOL

    # ry(π) sends Z → -Z as well (R_y(π) Z R_y(-π) = -Z).
    c2 = Circuit(1)
    c2.append(gates.ry(math.pi, 0))
    out2 = s.propagate(c2)
    dom2 = dominant_coeff(out2.coefficients())
    assert abs(dom2 - (-1 + 0j)) < TOL


def test_heisenberg_direction_reverses_a_pauli_rotation():
    # Forward rz(theta) then heisenberg rz(theta) on the same circuit returns
    # the input. This pins the adjoint and direction wiring.
    s = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.rz(0.7, 0)
    fwd = s.propagate(c)
    rev = fwd.propagate(c, direction="heisenberg")
    assert coeffs_close(rev.coefficients(), [1 + 0j])


def test_propagate_rejects_unknown_direction():
    s = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    c = Circuit(1)
    with pytest.raises(ValueError, match="forward"):
        s.propagate(c, direction="sideways")


def test_depolarize_method_appends_one_per_qubit():
    c = Circuit(3)
    c.depolarize(0.1, [0, 1, 2])
    assert len(c) == 3


def test_noise_factories_via_append():
    s = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    # Depolarizing with p=0 is the identity.
    c = Circuit(1)
    c.append(noise.depolarize(0.0, 0))
    out = s.propagate(c)
    assert coeffs_close(out.coefficients(), [1 + 0j])

    # Depolarizing on a Z with p=3/4 zeros out (1 - 4·(3/4)/3 = 0). Exact zero
    # is dropped by the merge.
    s_z = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    c2 = Circuit(1)
    c2.append(noise.depolarize(0.75, 0))
    out2 = s_z.propagate(c2)
    assert len(out2) == 0


def test_dephase_and_amplitude_damping_factories():
    # Dephasing(p=0.5) on X scales by 1 - 2·0.5 = 0 → dropped.
    s_x = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.append(noise.dephase(0.5, 0))
    out = s_x.propagate(c)
    assert len(out) == 0

    # AmplitudeDamping(γ=0) is the identity on Z.
    s_z = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    c2 = Circuit(1)
    c2.append(noise.amplitude_damping(0.0, 0))
    out2 = s_z.propagate(c2)
    assert coeffs_close(out2.coefficients(), [1 + 0j])
