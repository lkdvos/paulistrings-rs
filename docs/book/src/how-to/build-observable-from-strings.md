# Build an observable from Pauli strings

```python
from paulistrings import PauliSum

observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)
```

Each key is a Pauli string over `num_qubits` characters, qubit 0 first.
A coefficient multiplies the literal **Hermitian** Pauli string: `Y` carries no phase of its own, unlike a "canonical" phased Y elsewhere.
See [PauliSum reference](../reference/pauli-sum.md) for the full constructor surface and error cases (length mismatch, invalid character).
