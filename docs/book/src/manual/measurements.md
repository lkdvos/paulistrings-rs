# Measurements

## Product states {#product-states}

A propagated sum is read out with `expectation(state)`, which evaluates `⟨s|A|s⟩` for a single-qubit product state `|s⟩` in one masked pass over the terms.
The state is either a uniform shorthand — `"x+"`, `"y+"`, `"z+"`, the `+1` eigenstate of that Pauli on every qubit — or a per-qubit label string of exactly `num_qubits` characters from `0 1 + - r l`, qiskit's `Statevector.from_label` alphabet, with `0`/`1` the `Z` eigenstates, `+`/`-` the `X` ones and `r`/`l` the `Y` ones.
The default is `"x+"`, which is the state most pages on this site read against.

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)
circuit = Circuit(4)
circuit.rz(0.3, 0)
evolved = observable.propagate(circuit, truncation.coeff(1e-10), direction="heisenberg")

print(evolved.expectation("x+"))        # |++++>
print(evolved.expectation("z+").real)   # |0000>
print(evolved.expectation("0+1r").real) # |0>|+>|1>|+i> per qubit
```

```text
(0.9888341222814014+0j)
0.0
0.25
```

The return type is Python `complex`; take `.real` for a Hermitian observable, and treat a non-negligible imaginary part as a sign the sum is not Hermitian rather than as noise to discard.
A label of the wrong length, or a character outside the alphabet, is a `ValueError` that restates the rule.

The pass is cheap next to the propagation that produced the sum, and the state is the thing that is cheap to vary: one propagation, any number of product states.
The observable is not: the propagated object *is* the observable, so a second observable is a second propagation, not a second pass.
The label table is [Measurement](../library/measurement.md); [First propagation](../examples/first-propagation.md) is this readout on a full run.

## Which state the label denotes {#which-state}

`expectation` does not know which direction produced its sum, so the meaning of the label is fixed by the `direction` you passed to `propagate`.

| `direction` | `evolved.expectation(s)` equals | so `s` denotes |
|---|---|---|
| `"heisenberg"` | `⟨s\|U†OU\|s⟩`, the expectation of `O` in `U\|s⟩` | the **input** state, the one the circuit acts on |
| `"forward"` | `⟨s\|UOU†\|s⟩`, the expectation of `O` in `U†\|s⟩` | the state at the circuit's **output** end |

Swapping the two is silent: both return a number of the right size and type, and nothing raises.
One `rx` rotation on `Z`, read against the `Y` eigenstate `"r"`, shows the two answers differing by a sign:

```python
z0 = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
rotate = Circuit(1)
rotate.rx(0.7, 0)
print(z0.propagate(rotate, direction="heisenberg").expectation("r").real)
print(z0.propagate(rotate, direction="forward").expectation("r").real)
```

```text
0.644217687237691
-0.644217687237691
```

Most physics questions — a state prepared, a circuit run, an observable measured — are the Heisenberg row, and the label is the prepared state.
The worked example with both directions on the same circuit is [Direction](propagation/direction.md#which-state-the-label-is-read-against), and the table with the engine's view of each direction is [Direction semantics](../library/direction.md).

## Stabilizer states {#stabilizer-states}

A product-state label cannot express an entangled state; `expectation_stabilizer(generators)` reads the sum against any stabilizer state — Bell, GHZ, cluster, the output of any Clifford circuit — given as a list of exactly `num_qubits` signed generators.
Each is a Pauli string in the `from_strings` alphabet with an optional sign prefix, `"+XX"`, `"-ZZ"`, or bare `"ZZ"` for `+`; the generators must pairwise commute and be independent, and anything else is a `ValueError`.

```python
bell = PauliSum.from_strings({"YY": 1.0}, num_qubits=2)
print(bell.expectation_stabilizer(["XX", "ZZ"]).real)     # |Φ+>, YY = -XX·ZZ
print(bell.expectation_stabilizer(["-XX", "+ZZ"]).real)   # flip one sign, flip the state
```

```text
-1.0
1.0
```

Rather than write generators by hand, `interop.stabilizers_from_stim` reads them from a Clifford circuit's *output state* — a stim circuit, program text, tableau or tableau simulator — in the sign convention this library shares with stim.
It imports `stim` lazily on first call — install the `examples` or `bench` extra from [Installation](../installation.md) to use it.
`num_qubits=` pads the register on the right with `+Z` generators, leaving those qubits in `|0⟩`, so a stim circuit on a few qubits can be read against a wider sum.

```python
import stim
from paulistrings import interop

generators = interop.stabilizers_from_stim(stim.Circuit("H 0\nCNOT 0 1"))
print(generators, bell.expectation_stabilizer(generators).real)
print(interop.stabilizers_from_stim(stim.Circuit("H 0\nCNOT 0 1"), num_qubits=3))
```

```text
['+XX', '+ZZ'] -1.0
['+XXI', '+ZZI', '+IIZ']
```

The cost is `O(terms · num_qubits² / 64)` after a one-off `O(num_qubits³ / 64)` reduction of the generators, so a few times a product-state pass per term and still negligible next to the propagation.
This is the split [B7](../examples/showcases/b7-stabilizer-prep.md) is built on: stim prepares a 36-qubit cluster state in polynomial time, and Pauli propagation handles the non-Clifford tail it cannot touch.
Signatures and the generator rules are under [Measurement](../library/measurement.md).

## Overlaps and the identity coefficient {#overlap}

Two sums on the same register have a Hilbert–Schmidt overlap `a.overlap(b) = tr(a† b) / 2^n`, which is `Σ conj(a_P) b_P` over the strings they share, since distinct Pauli strings are orthogonal.
Its two everyday uses are the norm `Σ|c|²` as `a.overlap(a)`, introduced under [Inspecting a sum](operators.md#inspecting), and comparing two propagated results term for term without decoding either.
`identity_coefficient()` is the coefficient of `I…I`, or `0` when the sum has no identity term, which equals `tr(O) / 2^n` and hence the expectation of `O` in the maximally mixed state.

```python
truncated = observable.propagate(circuit, truncation.coeff(1e-1), direction="heisenberg")
print(evolved.overlap(evolved).real, truncated.overlap(evolved).real)
print(evolved.identity_coefficient())
```

```text
0.25 0.24454173796592743
0j
```

A `num_qubits` mismatch between the two sums is a `ValueError`; there is no implicit padding.
Both signatures are under [Measurement](../library/measurement.md#overlapother).

## From a number to a result

A single expectation value at a single cutoff is a number, not a result.
The truncation that made the sum tractable deleted terms carrying signs, so the value can sit on either side of the truth and need not move monotonically as the cutoff tightens — [No variational bound](propagation/truncation.md#no-variational-bound) is the statement and the measured cases.
What turns the number into a result is [Validating a result](propagation/validation.md): sweep the cutoff, read the retained norm alongside, apply the plateau criterion, and score against an exact reference wherever one is affordable.

## See it in use

- [First propagation](../examples/first-propagation.md#check-the-cutoff) — `expectation("x+")` after a Heisenberg run, then the same value across three cutoffs with the retained norm alongside.
- [B7 — Stabilizer-state preparation](../examples/showcases/b7-stabilizer-prep.md) — `stabilizers_from_stim` on a 36-qubit cluster state, `expectation_stabilizer` on the propagated observable.
- [B1 — Operator scrambling](../examples/showcases/b1-operator-scrambling.md#running-it) — the OTOC and the conserved norm computed from the coefficient arrays rather than an expectation call, the array route of [Symplectic arrays](operators.md#symplectic-arrays).
- [B2 — Noisy circuit verification](../examples/showcases/b2-noisy-verification.md#the-same-collapse-three-other-channels) — one `Z` expectation under four noise channels, and why amplitude damping moves it where the Pauli channels only shrink it.
