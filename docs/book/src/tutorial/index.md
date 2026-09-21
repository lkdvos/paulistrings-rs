# Your first propagation

This walks through one Heisenberg-picture propagation end to end: build an observable, build a circuit, propagate it, read out an expectation value.

```python
import math
from paulistrings import Circuit, PauliSum, truncation

# Observable: average X magnetization on 4 qubits. Coefficients multiply the
# literal Hermitian Pauli string — `Y` carries no phase of its own.
observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)

circuit = Circuit(4)
circuit.rz(math.pi / 8, 0)
circuit.cnot(0, 1)
circuit.h(2)

evolved = observable.propagate(
    circuit,
    truncation.coeff(1e-10),      # drop |c| <= 1e-10 after every channel
    direction="heisenberg",       # U† O U — always pass this explicitly
)
print(len(evolved), evolved.expectation("x+").real)
```

## Step by step

`PauliSum.from_strings` builds the observable directly from a dict of Pauli strings to coefficients.
Here it is the average X magnetization on 4 qubits: a quarter weight on each single-qubit `X` string.

`Circuit(4)` starts an empty 4-qubit circuit.
`circuit.rz(math.pi / 8, 0)` pushes a Z rotation on qubit 0.
`circuit.cnot(0, 1)` pushes a CNOT entangling qubits 0 and 1.
`circuit.h(2)` pushes a Hadamard on qubit 2.
Each call appends one gate, so the circuit is exactly the three-gate sequence written above, in that order.

`observable.propagate(...)` evolves the observable through the circuit.
`truncation.coeff(1e-10)` is the truncation policy: after every channel, drop any term with `|c| <= 1e-10`.
`direction="heisenberg"` says to compute `U† O U`, walking the circuit in reverse and applying each channel's adjoint — pass this explicitly, since the two directions answer different questions.

`evolved` is the propagated `PauliSum`.
`len(evolved)` is the number of surviving terms.
`evolved.expectation("x+")` evaluates the expectation value in the uniform product state `|+…+⟩`; `.real` drops the (here zero) imaginary part.

## See also

- `examples/common/observables.py` — more observable constructions.
- `examples/common/circuits.py` — more circuit constructions.
