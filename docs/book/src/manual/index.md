# Manual

The Manual is the reading path through the library: what each object is, how to use it, and where it bites.
Every computation here has the same shape: write an observable, build a circuit, propagate the observable through the circuit, measure the result against a state, then check that the truncation did not decide the answer.

```python
import math
from paulistrings import Circuit, PauliSum, pt

observable = PauliSum(4, pt("X", range(4)) / 4)

circuit = Circuit(4)
circuit.rz(math.pi / 8, 0)
circuit.cnot(0, 1)
circuit.h(2)

evolved = observable.propagate(circuit, direction="heisenberg")
print(len(evolved), evolved.expectation("x+").real)
```

![Circuit diagram: Rz(pi/8) on qubit 0, then a CNOT from qubit 0 to qubit 1, then H on qubit 2; qubit 3 is idle](../assets/quickstart/circuit.svg)

```text
5 0.7309698831278217
```

`pt` writes the observable $\frac{1}{4}\sum_j X_j$ and `PauliSum(4, ...)` binds it to four qubits.
The three gates conjugate it in reverse under `direction="heisenberg"`, and `expectation` reads the result against $|{+}{+}{+}{+}\rangle$.
Each step has a chapter, in this order:

- [Operators](operators.md): writing, storing, inspecting and saving an observable or Hamiltonian.
- [Circuits](circuits.md): gates, noise channels, Trotter steps, composition, and importing from stim, qiskit and task JSON.
- [Engine and propagation](propagation/index.md): the `propagate` call, direction, truncation, validation, and scaling out.
- [Measurements](measurements.md): reading a propagated sum against a product state, a stabilizer state, or another sum.

Read the chapters in order once; each assumes the ones before it.
Afterwards, enter by topic from the sidebar.
The [Library](../library/index.md) holds the exact signatures, and [Against other tools](../examples/comparisons.md#which-method-fits-which-problem) says when Pauli propagation is the right method at all.