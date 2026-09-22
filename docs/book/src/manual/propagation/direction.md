# Direction

`direction` selects which conjugation `propagate` performs, and the two choices compute different operators.
Under `"heisenberg"` the evolved sum is `U†OU`; under `"forward"` it is `UOU†`.
Both are valid, both are cheap or expensive for their own reasons, and neither raises when it is the wrong one for the question being asked — which is why this is the first page of the chapter.

**Heisenberg** answers "what does this circuit measure?" — you have an observable `O` and a circuit `U`, and want `⟨ψ|U†OU|ψ⟩` for a product state `|ψ⟩` the circuit acts on.
A local observable starts as a handful of terms and only spreads as far as its causal cone, which is why this is the direction almost every example on this site uses.

**Forward** answers "what does this circuit do to this operator?" — evolving a density matrix or a Hamiltonian in the Schrödinger picture.
It starts from a wide operator, so the cost profile is different from the outset.

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4)
circuit = Circuit(4)
circuit.rz(0.3, 0)
circuit.cnot(0, 1)
policy = truncation.coeff(1e-10)

heisenberg = observable.propagate(circuit, policy, direction="heisenberg")
forward = observable.propagate(circuit, policy, direction="forward")
```

**The default is `"forward"`, and most examples on this site do not use it.**
The Python binding maps `direction=None` to `"forward"`, so a call that omits the argument silently computes `UOU†`.
Treat the default as a one-off exploration convenience, not an idiom, and pass `direction` explicitly in anything you keep.

## Which state the label is read against {#which-state-the-label-is-read-against}

`expectation(state)` evaluates `⟨s|A|s⟩` for whatever sum `A` it is handed, so `direction` decides what the label `s` refers to:

- under `"heisenberg"` the evolved sum is `U†OU`, so `expectation(s)` is `O`'s expectation in `U|s⟩` — **`s` is the input state, the one the circuit acts on**;
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

The rule of thumb: decide which end of the circuit your state lives at, and that decides the direction.
A state prepared *before* the circuit is a Heisenberg run; an operator you want to see *after* the circuit has acted on it is a forward run.
[Measurements](../measurements.md#which-state) restates this from the measurement side, for readers who arrive there first.

## Push order {#push-order}

Under `"heisenberg"` the engine walks the channel list **in reverse** and applies each channel's adjoint; under `"forward"` it walks the list as written and applies each channel.
So pushing `ZZ` rotations before `X` rotations builds `U = U_X · U_ZZ`, and Heisenberg evolution computes `U_ZZ† U_X† O U_X U_ZZ` — the `X` rotations act on `O` first.
Build the circuit in the order the gates act on the *state*; the direction flag does the rest.

Two places this rule surfaces later in the chapter.
`PropagationStats` lists its layers in application order, so under `"heisenberg"` the first entry is the circuit's last gate — `circuit_index` recovers the position as written ([Stats](settings.md#stats)).
And an incremental propagation extends the evolved operator at the *application* end, which under `"heisenberg"` means a second `propagate` call prepends its circuit rather than appending it — [The Heisenberg-ordering trap](incremental.md#the-heisenberg-ordering-trap).

The full two-row table is in the [Library](../../library/direction.md).
Every cross-engine comparison on this site is run in the Heisenberg direction, because `PauliPropagation.jl` defines no forward map for several of the gates in the shared vocabulary — [Against other tools](../../examples/comparisons.md#methodology) has the asymmetry.
[Showcase B5](../../examples/showcases/b5-operator-backpropagation.md) is a Heisenberg run whose evolved observable is then handed to a shorter front circuit, which only works because the label denotes the input state.
