# Choose a propagation direction

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4)
circuit = Circuit(4)
circuit.rz(0.3, 0)
circuit.cnot(0, 1)
policy = truncation.coeff(1e-10)

evolved = observable.propagate(circuit, policy, direction="heisenberg")
```

**Heisenberg** for "what does this circuit measure?" — you have an observable and a circuit, and want `⟨ψ|U†OU|ψ⟩` for a product state `|ψ⟩`.
A local observable starts as a handful of terms and only spreads as far as its causal cone, which is why this is the direction almost every example on this site uses.

**Forward** for "what does this circuit do to this operator?" — evolving a density matrix or a Hamiltonian in the Schrödinger picture.
It starts from a wide operator, so the cost profile is different from the outset.

```python
evolved = observable.propagate(circuit, policy, direction="forward")
```

**Always pass `direction` explicitly.** The Python binding defaults `direction=None` to `"forward"`, but the two pictures answer different questions — treat the default as a one-off exploration convenience, not an idiom.

Push order interacts with direction: under `"heisenberg"` the engine walks the channel list **in reverse** and applies each channel's adjoint, so pushing `ZZ` rotations before `X` rotations builds `U = U_X · U_ZZ` and Heisenberg evolution computes `U_ZZ† U_X† O U_X U_ZZ`.
Get the circuit's push order right and the direction flag does the rest.

See [Direction reference](../reference/direction.md) for the full table and [Comparisons](../explanation/comparisons.md) for the cross-engine asymmetry in what `"forward"` can express.
