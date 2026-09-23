# Circuits

## Building a circuit

A `Circuit` is an ordered list of channels on a fixed register, and every method call on it pushes exactly one channel.
`Circuit(n)` is the empty circuit; the named gates are `h`, `s`, `sdg`, `x`, `y`, `z`, `cnot`, `cz`, `swap`, the rotations `rx`, `ry`, `rz` and `pauli_rotation`, and the checked matrices `unitary_1q` and `unitary_2q`.

Every rotation implements `exp(-i·θ·P/2)` for its generator `P`, the same convention as qiskit's `rz`.
A Trotter step written as `exp(-i·s·P)`, with no `/2`, therefore needs `theta = 2*s`; getting this wrong halves every angle and the run still completes, just for a different Hamiltonian.
`pauli_rotation(pauli, qubits, theta)` rotates about a Pauli string of any weight: `pauli[k]` acts on `qubits[k]`, and identity positions are left out rather than written as `I`, so `pauli_rotation("ZZ", [0, 1], theta)` is the two-qubit `ZZ` rotation.

```python
import math
from paulistrings import Circuit

n, J, h, dt = 4, 1.0, 0.5, 0.05
step = Circuit(n)
for i in range(n - 1):
    step.pauli_rotation("ZZ", [i, i + 1], theta=2 * -J * dt)   # exp(-i * (-J dt) * ZZ)
for i in range(n):
    step.rx(2 * -h * dt, i)                                     # exp(-i * (-h dt) * X)
print(len(step), step.gates[0])
```

```text
7 {'name': 'pauli_rotation', 'qubits': [0, 1], 'pauli': 'ZZ', 'theta': -0.1}
```

`circuit.gates` lists the channels as plain dicts, the task-JSON gate vocabulary, and is the quickest way to check what was actually pushed.
`unitary_1q(qubit, matrix)` and `unitary_2q(q0, q1, matrix)` take a `2x2` or `4x4` complex matrix, reject a non-unitary one with `ValueError`, and order the two-qubit tensor factors as `|q0 q1⟩`, `q0` more significant — so a CNOT matrix in the textbook basis is `unitary_2q(control, target, CNOT)`.
The same gates exist as factories in the `gates` module, returning a channel for `circuit.append`, for building a channel once and pushing it onto several circuits; `circuit.append(gates.h(0))` is `circuit.h(0)`.

The full table, with per-gate caveats, is [Circuit](../library/circuit.md#gates); [C — Deep Trotter circuits](../examples/benchmarks/c-deep-trotter.md) is the same shape of step — `ZZ` rotations then `X` kicks — on a 127-qubit heavy-hex lattice, 5 to 20 steps deep.

## Composition {#composition}

Circuits compose by channel list.
`a + b` is a new circuit with `a`'s channels then `b`'s, `a.extend(b)` appends in place, both require equal `num_qubits`, and `len`, indexing and slicing work on the channel list.
`circuit.adjoint()` is a new circuit with the channel order reversed and every gate replaced by its dagger: rotations negate `theta`, `s` becomes `sdg` and back, matrices are conjugate-transposed, and the self-adjoint Cliffords are unchanged.

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

The adjoint is what turns one direction into the other: propagating through `c` in the Heisenberg picture agrees, to floating-point tolerance, with propagating through `c.adjoint()` in the forward picture.
It exists only for unitary circuits — a noise channel has no adjoint in the time-reversal sense, so `adjoint()` on a circuit containing one raises `ValueError` naming the channel; take the adjoint of the unitary part and add the noise to the reversed circuit explicitly.
How push order and direction interact is [Push order](propagation/direction.md#push-order); the composition table is under [Circuit](../library/circuit.md#composition).

## One gate, one truncation point {#one-gate-one-channel}

Throughout this Manual, a **layer** is one applied channel — one gate, or one noise channel on one qubit or pair — never a brickwork layer of parallel gates.
Truncation runs after every layer in that sense, so `len(circuit)` is the number of truncation points a propagation will pass, and the per-layer records in `PropagationStats` have one entry per channel.

Two consequences follow.
Fusing two gates into one `unitary_2q` changes the answer, because it removes a truncation point between them; every circuit in the example suite is built one gate per push for that reason, and it is what makes a per-layer comparison against another engine meaningful at all.
And a broadcast call is still one channel per target: `depolarize(p, [0, 1])` pushes two channels, not one bundled one.

```python
noisy = Circuit(3)
noisy.h(0)
noisy.depolarize(0.01, [0, 1, 2])
print(len(noisy))
```

```text
4
```

Why a truncation point between gates matters, and why noise makes a fixed threshold bite harder rather than softer, is [A gate is a truncation point](propagation/truncation.md#a-gate-is-a-truncation-point).

## Noise channels {#noise-channels}

Noise is pushed like a gate, one channel per qubit in the list, with the `noise` module offering the same channels as factories for `append`.

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

In the Pauli basis a Pauli channel is a coefficient rescale, not a fan-out: `depolarize(p)` multiplies every non-identity Pauli on its qubit by `1 - 4p/3`, and `dephase(p)` leaves `Z` alone and scales `X` and `Y` by `1 - 2p`.
`pauli_channel(p/3, p/3, p/3, q)` is `depolarize(p, q)` and `pauli_channel(0, 0, p, q)` is `dephase(p, q)`.
`amplitude_damping` is the one channel here that is not a Pauli channel: it moves `Z` partly onto the identity, so it changes the key set, and it is not self-adjoint, which is why its Heisenberg and forward results differ in more than a sign.

Every noise channel is non-unitary, so a circuit that carries one has no `adjoint()`.
The rescale is why noise makes a simulation cheaper rather than dearer under a coefficient threshold, the effect [B2](../examples/showcases/b2-noisy-verification.md) measures; the semantics table is [Circuit](../library/circuit.md#noise-channels).

## Importing circuits {#importing-circuits}

Three importers in the `interop` module build a `Circuit` from an outside description.
Each maps what it can onto the gate vocabulary above and raises `ValueError` naming the instruction on anything it cannot; nothing is ever skipped silently.
`circuit_from_stim` and `circuit_from_qiskit` import `stim`/`qiskit` lazily on first call — install the `examples` or `bench` extra from [Installation](../installation.md) to use them.

`circuit_from_stim` takes a stim circuit object, stim program text, or a path to a `.stim` file, expands `REPEAT` blocks, and returns `(circuit, observable)`, where the observable comes from `OBSERVABLE_INCLUDE` instructions with Pauli targets or is `None`.
Measurements, resets, detectors and sweep or record targets are errors.

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

`circuit_from_qiskit` maps the named gates `h s sdg x y z cx cz swap rz rx ry` directly, `rzz`/`rxx`/`ryy` to `pauli_rotation`, and anything else with a one- or two-qubit unitary through the checked `unitary_1q`/`unitary_2q` path, reordering the two-qubit basis from qiskit's little-endian convention for you.
Measurements, resets, conditioned instructions and gates on more than two qubits are errors; qubit indices come from `find_bit`, so multi-register circuits import correctly.

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

`circuit_from_json` builds from the frozen task-JSON schema's `"circuit"` object, the interchange format shared with the cross-engine comparison, and `load_task` reads a whole task file — circuit, observable, truncation policy and direction — into one object.

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

Unknown keys, unknown gate names and a missing `run.direction` are errors.
Importer signatures are under [Circuit](../library/circuit.md#importing-a-circuit); the stim importer's sibling, `stabilizers_from_stim`, reads a Clifford circuit's *output state* rather than its gates and belongs to [Stabilizer states](measurements.md#stabilizer-states).

## Hard edge: support on more than two qubits

The engine applies a channel through a lookup on its support, and that support is capped at two qubits: a channel acting on three or more makes `propagate` **panic**, with no fallback path.
`pauli_rotation` is the one exemption and accepts a generator of any weight, so a `ZZZ` or weight-ten rotation is fine.

From Python the constructors keep you inside the edge: every named gate and noise method is one- or two-qubit, `unitary_2q` is the widest matrix accepted, and the importers reject a three-qubit instruction with `ValueError` before it reaches a circuit.
The edge is therefore what a three-qubit gate becomes when you need one: a decomposition into one- and two-qubit gates, each its own truncation point, or a Pauli rotation if it is one.

## See it in use

- [B2 — Noisy circuit verification](../examples/showcases/b2-noisy-verification.md) — a depolarizing channel on every qubit a gate just touched, at 127 qubits, then the same run with each of the other channels swapped in, and the collapse in term count that follows.
- [B5 — Hybrid depth reduction](../examples/showcases/b5-operator-backpropagation.md#running-it) — a circuit split into a front and a tail, the tail's propagated observable written to a task JSON and read back with `load_task`.
- [A — Clifford point](../examples/benchmarks/a-clifford.md) — the same recorded gate list handed to the engine and, as schema-v1 task JSON, to `PauliPropagation.jl`, for a term-for-term parity check.
- [C — Deep Trotter circuits](../examples/benchmarks/c-deep-trotter.md) — a kicked-Ising Trotter step on 127 qubits, one rotation per push, 5 420 channels at 20 steps.
