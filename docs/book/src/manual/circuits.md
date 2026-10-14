# Circuits

Once we have an operator, we need a circuit to propagate it through.
A circuit is the ordered list of gates and noise channels applied to a register.
This chapter shows how to build one gate by gate or from a Hamiltonian, how to combine circuits, which noise channels are available, and how to import circuits from stim, qiskit and task JSON.

## Building a circuit

A `Circuit` is an ordered list of channels on a fixed register.
`Circuit(n)` is the empty circuit; the gates are `h`, `s`, `sdg`, `x`, `y`, `z`, `cnot`, `cz`, `swap`, the rotations `rx`, `ry`, `rz` and `pauli_rotation`, and the checked matrices `unitary_1q` and `unitary_2q`.

Every rotation implements $R_P(\theta) = \exp(-i\theta P/2)$ for its generator $P$, the same convention as qiskit's `rz`.
`pauli_rotation(pauli, qubits, theta)` rotates about a Pauli string of any weight: `pauli[k]` acts on `qubits[k]`, and identity positions are left out rather than written as `I`.

The single-qubit gates and the noise methods take one qubit or a list of qubits, and push one channel per qubit.
`pauli_rotation` takes one tuple of sites or a list of tuples, and a one-letter label with a list of sites gives one rotation per site.

```python
from paulistrings import Circuit
from paulistrings.lattices import chain

step = Circuit(4)
step.pauli_rotation("ZZ", chain(4), 0.2)
step.rx(0.3, range(4))
print(len(step), step.gates[0])
```

```text
7 {'name': 'pauli_rotation', 'qubits': [0, 1], 'pauli': 'ZZ', 'theta': 0.2}
```

`circuit.gates` lists the channels as plain dicts, which is the quickest way to check what was pushed.
A call that names a qubit outside the register raises `ValueError` and pushes nothing:

```python
try:
    step.depolarize(0.01, [0, 9])
except ValueError as err:
    print(err)
print(len(step))
```

```text
qubit index 9 is out of range for a 4-qubit circuit
7
```

`unitary_1q(qubit, matrix)` and `unitary_2q(q0, q1, matrix)` take a `2x2` or `4x4` complex matrix, reject a non-unitary one with `ValueError`, and order the two-qubit tensor factors as $|q_0 q_1\rangle$, so a CNOT matrix in the textbook basis is `unitary_2q(control, target, CNOT)`.
The `gates` and `noise` modules provide the same channels as factories for `circuit.append`, one qubit at a time: `circuit.append(gates.h(0))` is `circuit.h(0)`.
The full table is in [Circuit](../library/circuit.md#gates).

## Trotter steps {#trotter-steps}

For a Hamiltonian $H = \sum_k h_k Q_k$ written as `PauliTerms` ([Sparse terms](operators.md#sparse-terms)), `trotter(H, dt, num_qubits)` returns one first-order Trotter step of $e^{-iH\,\delta t}$.
It pushes a `pauli_rotation` about each $Q_k$ with angle $\theta = 2h_k\,\delta t$, in the order the terms were written.

```python
from paulistrings import pt, trotter

L, J, h, dt = 4, 1.0, 0.5, 0.05
H = -J * pt("ZZ", chain(L)) - h * pt("X", range(L))
step = trotter(H, dt, L)
print(len(step), len(trotter(H, dt, L, order=2)), step.gates[0])
```

```text
7 13 {'name': 'pauli_rotation', 'qubits': [0, 1], 'pauli': 'ZZ', 'theta': -0.1}
```

The factor of 2 comes from the rotation convention, and the sign of $h_k$ goes into the angle, so the $ZZ$ term with $h_k = -J$ gets $\theta = -2J\,\delta t$.
`order=2` gives the symmetric step: every term but the last at $\delta t/2$, the last at $\delta t$, then the others in reverse order at $\delta t/2$.
`steps=k` repeats the step $k$ times, and `circuit.trotter(H, dt, order)` appends one step to an existing circuit.
Every coefficient must be real.

The error of a first-order step comes from the commutators $[Q_j, Q_k]$, so terms that all commute give an exact step.
Otherwise $\delta t$ is a second approximation next to the truncation, and a result has to be converged in both ([Validating a result](propagation/validation.md)).

## Composition and layers {#composition}

What if you need to compose circuits?
`a + b` is a new circuit with `a`'s channels then `b`'s, `a.extend(b)` appends in place, both require equal `num_qubits`, and `len`, indexing and slicing work on the channel list.
`circuit.adjoint()` reverses the channel order and replaces every gate by its dagger: rotations negate `theta`, `s` becomes `sdg` and back, matrices are conjugate-transposed, and the self-adjoint Cliffords are unchanged.

```python
front = Circuit(2)
front.h(0)
front.s(0)
back = Circuit(2)
back.rz(0.3, 1)

whole = front + back
print(len(front), len(back), len(whole))
print([g["name"] for g in whole.adjoint().gates], whole.adjoint().gates[0]["theta"])
```

```text
2 1 3
['rz', 'sdg', 'h'] -0.3
```

Propagating through `c` in the Heisenberg picture agrees, to floating-point tolerance, with propagating through `c.adjoint()` in the forward picture.
A noise channel has no adjoint, so `adjoint()` on a circuit containing one raises `ValueError` naming the channel.
How push order and direction interact is [given here](propagation/direction.md#push-order).

<a id="one-gate-one-channel"></a>
Every channel in the list is one gate, or one noise channel on one qubit or pair.
We call each applied channel a layer; this is not a brickwork layer of parallel gates.
Truncation runs after every layer, so `len(circuit)` is the number of truncation points a propagation passes, and `PropagationStats` has one entry per channel.
For the same reason, fusing two gates into one `unitary_2q` changes the answer, since it removes the truncation point between them.
A broadcast call or a Trotter step still adds one channel per target or term:

```python
noisy = Circuit(3)
noisy.h(0)
noisy.depolarize(0.01, [0, 1, 2])
print(len(noisy))
```

```text
4
```

Why a truncation point between gates matters is explained in [A gate is a truncation point](propagation/truncation.md#a-gate-is-a-truncation-point).

## Noise channels {#noise-channels}

Noise is pushed like a gate, one channel per qubit in the list.

```python
from paulistrings import noise

circuit = Circuit(4)
circuit.h(0)
circuit.depolarize(0.01, [0])                    # p spread evenly over X, Y, Z
circuit.cnot(0, 1)
circuit.depolarize2(0.01, [(0, 1)])              # p over the 15 non-identity two-qubit Paulis
circuit.dephase(0.01, [1])                       # Z error with probability p
circuit.amplitude_damping(0.01, [2])             # relaxation toward |0> with probability gamma
circuit.pauli_channel(0.002, 0.002, 0.008, [3])  # X, Y, Z with px, py, pz; px + py + pz <= 1
circuit.append(noise.depolarize(0.01, qubit=0))
print(len(circuit))
```

```text
8
```

In the Pauli basis a Pauli channel rescales coefficients: `depolarize(p)` multiplies every non-identity Pauli on its qubit by $1 - 4p/3$, and `dephase(p)` leaves $Z$ alone and scales $X$ and $Y$ by $1 - 2p$.
`pauli_channel(p/3, p/3, p/3, q)` is `depolarize(p, q)`, and `pauli_channel(0, 0, p, q)` is `dephase(p, q)`.
`amplitude_damping` is not a Pauli channel: it moves $Z$ partly onto the identity, and its Heisenberg and forward results differ.
The semantics table is in [Circuit](../library/circuit.md#noise-channels).

## Importing circuits {#importing-circuits}

Three importers in the `interop` module build a `Circuit` from an outside description, and raise `ValueError` naming any instruction they cannot map.
`circuit_from_stim` and `circuit_from_qiskit` need the `examples` or `bench` extra from [Installation](../installation.md).

`circuit_from_stim` takes a stim circuit, program text, or a path to a `.stim` file, expands `REPEAT` blocks, and returns `(circuit, observable)`, where the observable comes from `OBSERVABLE_INCLUDE` instructions or is `None`.

```python
import stim
from paulistrings import interop

circuit, observable = interop.circuit_from_stim(
    stim.Circuit("H 0\nCNOT 0 1\nDEPOLARIZE1(0.01) 0 1\nOBSERVABLE_INCLUDE(0) Z0 Z1")
)
print(len(circuit), [g["name"] for g in circuit.gates], observable.coefficients())
```

```text
4 ['h', 'cnot', 'depolarize', 'depolarize'] [(1+0j)]
```

`circuit_from_qiskit` maps `h s sdg x y z cx cz swap rz rx ry` directly, `rzz`/`rxx`/`ryy` to `pauli_rotation`, and any other one- or two-qubit gate through `unitary_1q`/`unitary_2q`, converting qiskit's little-endian two-qubit basis.

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.rzz(0.3, 0, 1)
qc.t(1)
print([g["name"] for g in interop.circuit_from_qiskit(qc).gates])
```

```text
['h', 'cnot', 'pauli_rotation', 'unitary_1q']
```

`circuit_from_json` builds from the task-JSON schema's `"circuit"` object, and `load_task` reads a whole task file (circuit, observable, truncation policy and direction):

```python
task = interop.load_task({
    "version": 1,
    "n_qubits": 2,
    "circuit": {"gates": [{"name": "h", "qubits": [0]}, {"name": "rz", "qubits": [1], "theta": 0.3}]},
    "observable": {"ZI": 1.0},
    "truncation": {"min_abs_coeff": 1e-8},
    "run": {"direction": "heisenberg"},
})
print(len(task.circuit), len(task.observable), task.direction)
```

```text
2 1 heisenberg
```

Importer signatures are in [Circuit](../library/circuit.md#importing-a-circuit).

## Support on more than two qubits

The engine applies a channel through a lookup on its support, capped at two qubits.
`pauli_rotation` is the exception and accepts a generator of any weight.

## See it in use

- [B2 — Noisy circuit verification](../examples/showcases/b2-noisy-verification.md): depolarizing noise on a 127-qubit circuit.
- [B5 — Hybrid depth reduction](../examples/showcases/b5-operator-backpropagation.md#running-it): a circuit split into a front and a tail, the tail read back with `load_task`.
- [A — Clifford point](../examples/benchmarks/a-clifford.md): one gate list run here and in `PauliPropagation.jl`.
- [C — Deep Trotter circuits](../examples/benchmarks/c-deep-trotter.md): a kicked-Ising step on 127 qubits.