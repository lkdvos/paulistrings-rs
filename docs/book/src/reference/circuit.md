# Circuit

## Constructor

`Circuit(num_qubits)` — an empty circuit.

## Conventions

**One gate per channel.** Every `Circuit` method pushes exactly one channel per qubit/pair — a broadcast call like `depolarize(p, [0, 1])` pushes two channels, not one bundled channel.
Truncation runs after every channel, never after a bundle, so fusing gates changes the answer; build circuits one gate per call.

**Rotation angle convention.** Every rotation method implements `U = exp(-i*theta*P/2)` for its generator `P`.
A generator coefficient `s` in `exp(-i*s*P)` (no `/2`) needs `theta = 2*s`.

## Gates

Each of these is a `Circuit` method, a `gates.<name>(...)` factory returning a `Channel` for `circuit.append(...)`, or both:

| `Circuit` method | `gates.` factory | Qubits | Extra args |
|---|---|---|---|
| `.h(qubit)` | `gates.h(qubit)` | 1 | — |
| `.s(qubit)` | `gates.s(qubit)` | 1 | — |
| `.sdg(qubit)` | `gates.sdg(qubit)` | 1 | `S^dagger` |
| `.x(qubit)` | `gates.x(qubit)` | 1 | — |
| `.y(qubit)` | `gates.y(qubit)` | 1 | — |
| `.z(qubit)` | `gates.z(qubit)` | 1 | — |
| `.cnot(control, target)` | `gates.cnot(control, target)` | 2 | indices must differ |
| `.cz(q0, q1)` | `gates.cz(q0, q1)` | 2 | indices must differ |
| `.swap(q0, q1)` | `gates.swap(q0, q1)` | 2 | indices must differ |
| `.rz(theta, qubit)` | `gates.rz(theta, qubit)` | 1 | `theta` |
| `.rx(theta, qubit)` | `gates.rx(theta, qubit)` | 1 | `theta` |
| `.ry(theta, qubit)` | `gates.ry(theta, qubit)` | 1 | `theta` |
| `.pauli_rotation(pauli, qubits, theta)` | `gates.pauli_rotation(pauli, qubits, theta)` | any | `pauli[k]` acts on `qubits[k]`; identity positions are expressed by omission, not `I` |
| `.unitary_1q(qubit, matrix)` | `gates.unitary_1q(qubit, matrix)` | 1 | `2x2` complex, checked unitary |
| `.unitary_2q(q0, q1, matrix)` | `gates.unitary_2q(q0, q1, matrix)` | 2 | `4x4` complex, checked unitary; `q0` is the more significant tensor factor (`\|q0 q1>`) |

`circuit.append(gates.h(0))` is equivalent to `circuit.h(0)`.

## Noise channels

| `Circuit` method | `noise.` factory | Semantics |
|---|---|---|
| `.depolarize(p, qubits)` | `noise.depolarize(p, qubit)` | one channel per qubit in `qubits` |
| `.dephase(p, qubits)` | `noise.dephase(p, qubit)` | one channel per qubit |
| `.amplitude_damping(gamma, qubits)` | `noise.amplitude_damping(gamma, qubit)` | one channel per qubit |
| `.pauli_channel(px, py, pz, qubits)` | `noise.pauli_channel(px, py, pz, qubit)` | `px + py + pz <= 1`; one channel per qubit |
| `.depolarize2(p, pairs)` | `noise.depolarize2(p, q0, q1)` | one channel per `(q0, q1)` pair; indices in a pair must differ |

`pauli_channel(p/3, p/3, p/3, q)` is `depolarize(p, q)`; `pauli_channel(0, 0, p, q)` is `dephase(p, q)`.
See [Circuit noise](../how-to/add-noise-to-a-circuit.md) for a worked example.

## Composition

| Operation | Effect |
|---|---|
| `len(circuit)` | channel count |
| `circuit[i]` | the channel at `i` (a `Channel`) |
| `circuit[a:b]` | a new `Circuit` of the selected channels, same width; every slice form works, including a negative step (which reverses channel order, not the adjoint) |
| `circuit.append(channel)` | push one `Channel` from a `gates`/`noise` factory |
| `a.extend(b)` | append every channel of `b` to `a`, in place; both must share `num_qubits` |
| `a + b` | new circuit: `a`'s channels then `b`'s; neither operand is modified |
| `circuit.adjoint()` | new circuit: reversed channel order, each gate replaced by its dagger |
| `circuit.gates` | the channel list as a list of task-JSON schema-v1 gate dicts |

`circuit.adjoint()` satisfies `obs.propagate(c, direction="heisenberg") == obs.propagate(c.adjoint(), direction="forward")` to floating-point tolerance.
Per gate: `rz`/`rx`/`ry`/`pauli_rotation` negate `theta`, `s` becomes `sdg` (and back), `unitary_1q`/`unitary_2q` conjugate-transpose, and `h`/`x`/`y`/`z`/`cnot`/`cz`/`swap` are self-adjoint.
A non-unitary channel (a noise channel) raises `ValueError` naming it.

## Importing a circuit

| Call | Source |
|---|---|
| `paulistrings.interop.circuit_from_stim(src)` | a stim circuit; returns `(Circuit, PauliSum \| None)` |
| `paulistrings.interop.circuit_from_qiskit(qc)` | a `qiskit.QuantumCircuit` |
| `paulistrings.interop.circuit_from_json(obj, n_qubits)` | the task-JSON schema-v1 `"circuit"` object |

See [stim and qiskit circuit import](../how-to/import-circuit-from-stim-or-qiskit.md) for the recipe.
