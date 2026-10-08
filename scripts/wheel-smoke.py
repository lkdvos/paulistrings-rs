"""Import an installed paulistrings wheel and check one hand-computed propagation per width tier."""

from paulistrings import Circuit, PauliSum

# H Z H = X, so Heisenberg-evolving Z_q under H_q gives X_q, whose |+> expectation is 1.
# 1, 100 and 1000 qubits exercise the W = 1, 2 and 16 monomorphizations.
for n in (1, 100, 1000):
    q = n - 1
    observable = PauliSum.from_strings({"I" * q + "Z": 1.0}, num_qubits=n)
    circuit = Circuit(n)
    circuit.h(q)
    value = observable.propagate(circuit, direction="heisenberg").expectation("x+")
    assert abs(value - 1.0) < 1e-12, (n, value)

print("paulistrings wheel smoke test passed")
