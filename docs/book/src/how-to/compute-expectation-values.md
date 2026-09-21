# Expectation values

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4)
circuit = Circuit(4)
circuit.rz(0.3, 0)
evolved = observable.propagate(circuit, truncation.coeff(1e-10), direction="heisenberg")

evolved.expectation("x+")      # uniform product state |+...+>
evolved.expectation("z+")      # |0...0>
evolved.expectation("0+1r")    # per-qubit label, qiskit's Statevector.from_label alphabet
```

`expectation(state)` takes either a uniform shorthand (`"x+"`, `"y+"`, `"z+"`) or a per-qubit label string, one character per qubit (`0 1 + - r l`), and defaults to `"x+"` when no state is given.

For a stabilizer state, pass its signed generators instead of a product-state label:

```python
bell = PauliSum.from_strings({"YY": 1.0}, num_qubits=2)
bell.expectation_stabilizer(["XX", "ZZ"])     # Bell state, generators default to "+"
bell.expectation_stabilizer(["-XX", "+ZZ"])   # explicit signs flip the group elements they touch
```

`interop.stabilizers_from_stim` builds that generator list from a Clifford circuit's *output state* rather than its gates:

```python
import stim
from paulistrings import interop

generators = interop.stabilizers_from_stim(stim.Circuit("H 0\nCNOT 0 1"))
bell.expectation_stabilizer(generators)
```

For an observable against another Pauli-sum state, use the Hilbert-Schmidt overlap instead:

```python
state = PauliSum.from_strings({"XIII": 1.0}, num_qubits=4)
observable.overlap(state)          # tr(A†B) / 2**n
observable.identity_coefficient()  # the coefficient of I...I
```

See [Measurement reference](../reference/measurement.md) for the full signatures and label alphabet.
