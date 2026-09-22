# First propagation

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

```text
5 0.7309698831278217
```

Five Pauli strings survived, and the average X magnetization in `|++++⟩` after the circuit is `0.731`.

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

## Look at the terms that survived

`PauliSum` has no term-listing accessor: it hands back the symplectic columns the engine stores, and a Pauli character is the pair of bits for that qubit.
Qubit `q`'s Pauli is bit `q` of the x word and bit `q` of the z word — `(0,0)` is `I`, `(1,0)` is `X`, `(0,1)` is `Z`, `(1,1)` is `Y`.
Decode the five terms and sort them by magnitude:

```python
import numpy as np

PAULI = np.array([["I", "Z"], ["X", "Y"]])

x = evolved.x_array()
z = evolved.z_array()
coefficients = evolved.coefficients_array()

for row in np.argsort(-np.abs(coefficients)):
    label = "".join(PAULI[(x[row, 0] >> q) & 1, (z[row, 0] >> q) & 1] for q in range(4))
    print(f"{label}  {coefficients[row].real:+.9f}")
```

```text
IIZI  +0.250000000
IXII  +0.250000000
IIIX  +0.250000000
XXII  +0.230969883
YXII  -0.095670858
```

Two of the four seed terms, `IXII` and `IIIX`, came through untouched, on the qubits the `cnot` and the `rz` never reach.
`IIXI` picked up `circuit.h(2)` and became `IIZI`, since a Hadamard swaps `X` and `Z` on the qubit it acts on.
`XIII` is the one the `cnot` and the `rz` both acted on, since qubit 0 is the CNOT's control: `cnot(0, 1)` turns it into `XXII`, and `rz(math.pi / 8, 0)` then splits that into `XXII` and `YXII`.
The `[row, 0]` index picks word 0 of the term, which is the whole key here because 4 qubits fit in one 64-bit word — see [Observable from raw arrays](../how-to/build-observable-from-arrays.md) for the general layout.

## Check the cutoff

`truncation.coeff(1e-10)` was a guess.
Run the same propagation at a few looser cutoffs and watch the term count, the answer and the retained norm `Σ|c|²` together:

```python
norm_in = float(np.sum(np.abs(observable.coefficients_array()) ** 2))

for eps in [1e-1, 1e-2, 1e-10]:
    trial = observable.propagate(circuit, truncation.coeff(eps), direction="heisenberg")
    retained = float(np.sum(np.abs(trial.coefficients_array()) ** 2)) / norm_in
    print(f"{eps:7.0e}  terms={len(trial):3d}  <O>={trial.expectation('x+').real:+.9f}  retained={retained:.6f}")
```

```text
  1e-01  terms=  4  <O>=+0.730969883  retained=0.963388
  1e-02  terms=  5  <O>=+0.730969883  retained=1.000000
  1e-10  terms=  5  <O>=+0.730969883  retained=1.000000
```

The answer is the same to every printed digit at all three cutoffs, and `1e-2` already retains the whole operator — this circuit is small enough that the truncation never bit.

Read the top row carefully, though.
At `1e-1` the sum lost 3.7% of its norm and the answer did not move at all, because the deleted `YXII` term has a `Y` on qubit 0 and contributes nothing to `⟨++++|O|++++⟩`.
A flat answer over a *changing* sum is not evidence of convergence, and on a real problem that is exactly the trap to check for — [Validate a result](../how-to/validate-a-result.md) is the full recipe.

## Where to go next

- [How-to guides](../how-to/index.md) — the task recipes: noise, truncation, interop, stats, NUMA, MPI.
- [Validate a result](../how-to/validate-a-result.md) — the cutoff sweep, the retained norm, and when a number may be quoted.
- [Propagation direction](../how-to/choose-propagation-direction.md) and [Direction semantics](../reference/direction.md) — what `"heisenberg"` and `"forward"` compute, and which state `expectation` is then read against.
- [Observable vs time](../how-to/propagate-a-time-series.md) — the same walkthrough, stepped out into a Trotterized time series.
- [Comparisons](../explanation/comparisons.md) — whether Pauli propagation is the right method for your problem at all.
- [`examples/common/observables.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/common/observables.py) and [`examples/common/circuits.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/common/circuits.py) — more observable and circuit constructions.
