# stim and qiskit circuit import

```python
import stim
from paulistrings import interop

circuit, observable = interop.circuit_from_stim(stim.Circuit("H 0\nCNOT 0 1"))
```

`circuit_from_stim` accepts a `stim.Circuit`, a path to a `.stim` file, or stim program text; `REPEAT` blocks are expanded before translation.
It returns `(circuit, observable)`, where `observable` is built from any `OBSERVABLE_INCLUDE` instructions, or `None` if there are none.
Anything it can't translate (measurements, resets, sweep/combined targets, unmapped noise instructions) is a hard `ValueError` naming the instruction — nothing is skipped silently.

```python
from qiskit import QuantumCircuit
from paulistrings import interop

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
circuit = interop.circuit_from_qiskit(qc)
```

`circuit_from_qiskit` maps named gates directly (`h s sdg x y z cx cz swap rz rx ry`, plus `rzz/rxx/ryy` to `pauli_rotation`) and falls back to `unitary_1q`/`unitary_2q` for anything else exposing a `qiskit.quantum_info.Operator`.
Measurements, resets, conditioned instructions, and gates on more than two qubits are hard errors.

A third path builds a `Circuit` from the frozen task-JSON schema, e.g. when reading a job file produced by the [cross-engine comparisons](../explanation/comparisons.md):

```python
circuit = interop.circuit_from_json({"gates": [{"name": "h", "qubits": [0]}]}, n_qubits=1)
```

See [Circuit reference](../reference/circuit.md) for the gate vocabulary each importer maps onto, and `interop.stabilizers_from_stim` in [Expectation values](compute-expectation-values.md) for reading out a Clifford circuit's output state rather than its gates.
