# Propagation direction

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

## Which state the label is read against

`expectation(state)` evaluates `⟨s|A|s⟩` for whatever sum it is handed, so `direction` decides what `s` refers to:

- under `"heisenberg"` the evolved sum is `U†OU`, so `expectation(s)` is `O`'s expectation in `U|s⟩` — **`s` is the input state, before the circuit**;
- under `"forward"` the evolved sum is `UOU†`, so `expectation(s)` is `O`'s expectation in `U†|s⟩` — **`s` is a state at the circuit's output end**.

Getting it backwards is silent.
The same observable, the same circuit and the same state label give two perfectly plausible numbers:

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"ZII": 1.0}, num_qubits=3)

both = Circuit(3)
both.h(0)
both.pauli_rotation("ZZ", [0, 1], 0.6)
both.rx(0.9, 0)
both.rx(0.9, 2)
both.cnot(1, 2)

exact = truncation.coeff(1e-12)
heisenberg = observable.propagate(both, exact, direction="heisenberg")
forward = observable.propagate(both, exact, direction="forward")

print("heisenberg", heisenberg.expectation("x+").real)
print("forward   ", forward.expectation("x+").real)
```

```text
heisenberg 0.6216099682706644
forward    0.8253356149096783
```

`0.6216…` is `⟨Z_0⟩` measured after running this circuit on `|+++⟩` — the Heisenberg answer, and the one nearly every question of the form "what does this circuit measure?" wants.
`0.8253…` is the same observable pushed forward through the circuit and then read against `|+++⟩` at the far end, which answers a different question.
Both were checked against a dense `2³×2³` construction; neither is a bug.

## Push order

Push order interacts with direction: under `"heisenberg"` the engine walks the channel list **in reverse** and applies each channel's adjoint, so pushing `ZZ` rotations before `X` rotations builds `U = U_X · U_ZZ` and Heisenberg evolution computes `U_ZZ† U_X† O U_X U_ZZ`.
Get the circuit's push order right and the direction flag does the rest.

Direction also decides which end an *incremental* propagation extends, which is what a time series is built from — see [Observable vs time](propagate-a-time-series.md).

See [Direction reference](../reference/direction.md) for the full table and [Comparisons](../explanation/comparisons.md) for the cross-engine asymmetry in what `"forward"` can express.
