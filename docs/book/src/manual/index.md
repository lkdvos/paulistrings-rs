# Manual

The Manual is the reading path through the library: what each object is, how to use it, and where it bites.
Every computation here has the same shape: write an observable, build a circuit, propagate the observable through the circuit, measure the result against a state, then validate that the truncation did not decide the answer.

```python
import math
from paulistrings import Circuit, PauliSum

observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)

circuit = Circuit(4)
circuit.rz(math.pi / 8, 0)
circuit.cnot(0, 1)
circuit.h(2)

evolved = observable.propagate(circuit, direction="heisenberg")
print(len(evolved), evolved.expectation("x+").real)
```

```text
5 0.7309698831278217
```

Four lines, four chapters: `PauliSum.from_strings` is [Operators](operators.md), `Circuit`/`rz`/`cnot`/`h` is [Circuits](circuits.md), `propagate(..., direction=...)` is [Engine and propagation](propagation/index.md), and `expectation(...)` is [Measurements](measurements.md).
The four chapters follow that same order.

- [Operators](operators.md) — the Pauli sum: how an observable or Hamiltonian is written, stored, inspected and saved.
- [Circuits](circuits.md) — gates and noise channels, composition, importing from stim, qiskit and task JSON, and why one gate is one truncation point.
- [Engine and propagation](propagation/index.md) — the `propagate` call: direction, truncation, validation, incremental runs, stats and memory, the engine internals, and scaling out over NUMA partitions and MPI ranks.
- [Measurements](measurements.md) — reading a propagated sum against a product state, a stabilizer state, or another Pauli sum, and which state the label denotes in each direction.

Read the chapters in order once; each assumes the ones before it and does not repeat them.
Afterwards, enter by topic from the sidebar: every section is self-contained enough to answer one question, and ends with a pointer to the [Library](../library/index.md) page holding the exact signatures and to an [Example](../examples/index.md) that uses the same call at scale.

- [Against other tools](../examples/comparisons.md#which-method-fits-which-problem) says whether Pauli propagation is the right method for your problem at all, and names the two places a state-vector or stabilizer simulator is strictly better.
- [Library](../library/index.md) is the signature-and-table reference the Manual cites throughout; it holds no rationale of its own.
- [First propagation](../examples/first-propagation.md) carries the run above further, into the surviving terms' identities and a cutoff-sweep validation.
