# Circuit

## Constructor

`Circuit(num_qubits)` — an empty circuit.

## Conventions

**One gate per channel.** Every `Circuit` method pushes exactly one channel per qubit/pair — a broadcast call like `depolarize(p, [0, 1])` pushes two channels, not one bundled channel.
Truncation runs after every channel, never after a bundle, so fusing gates changes the answer; build circuits one gate per call — see [Truncation](../manual/propagation/truncation.md).

**Rotation angle convention.** Every rotation method implements `U = exp(-i*theta*P/2)` for its generator `P`.
A generator coefficient `s` in `exp(-i*s*P)` (no `/2`) needs `theta = 2*s`.

## Gates

Each of these is a `Circuit` method, a `gates.<name>(...)` factory returning a `Channel` for `circuit.append(...)`, or both:

| `Circuit` method | `gates.` factory | Qubits | Extra args | Caveats |
|---|---|---|---|---|
| `.h(qubit)` | `gates.h(qubit)` | 1 | — | — |
| `.s(qubit)` | `gates.s(qubit)` | 1 | — | — |
| `.sdg(qubit)` | `gates.sdg(qubit)` | 1 | `S^dagger` | — |
| `.x(qubit)` | `gates.x(qubit)` | 1 | — | — |
| `.y(qubit)` | `gates.y(qubit)` | 1 | — | — |
| `.z(qubit)` | `gates.z(qubit)` | 1 | — | — |
| `.cnot(control, target)` | `gates.cnot(control, target)` | 2 | indices must differ | — |
| `.cz(q0, q1)` | `gates.cz(q0, q1)` | 2 | indices must differ | — |
| `.swap(q0, q1)` | `gates.swap(q0, q1)` | 2 | indices must differ | — |
| `.rz(theta, qubit)` | `gates.rz(theta, qubit)` | 1 | `theta` | — |
| `.rx(theta, qubit)` | `gates.rx(theta, qubit)` | 1 | `theta` | — |
| `.ry(theta, qubit)` | `gates.ry(theta, qubit)` | 1 | `theta` | — |
| `.pauli_rotation(pauli, qubits, theta)` | `gates.pauli_rotation(pauli, qubits, theta)` | any | `pauli[k]` acts on `qubits[k]`; identity positions are expressed by omission, not `I` | the one exemption from the two-qubit support panic below — any generator weight is accepted |
| `.unitary_1q(qubit, matrix)` | `gates.unitary_1q(qubit, matrix)` | 1 | `2x2` complex, checked unitary | — |
| `.unitary_2q(q0, q1, matrix)` | `gates.unitary_2q(q0, q1, matrix)` | 2 | `4x4` complex, checked unitary; `q0` is the more significant tensor factor (`\|q0 q1>`) | — |

A channel with support on more than two qubits, other than `pauli_rotation`, makes `propagate` **panic**; there is no fallback path.

`circuit.append(gates.h(0))` is equivalent to `circuit.h(0)`.

## Noise channels

| `Circuit` method | `noise.` factory | Semantics | Caveats |
|---|---|---|---|
| `.depolarize(p, qubits)` | `noise.depolarize(p, qubit)` | one channel per qubit in `qubits` | non-unitary; see below |
| `.dephase(p, qubits)` | `noise.dephase(p, qubit)` | one channel per qubit | non-unitary; see below |
| `.amplitude_damping(gamma, qubits)` | `noise.amplitude_damping(gamma, qubit)` | one channel per qubit | non-unitary; see below |
| `.pauli_channel(px, py, pz, qubits)` | `noise.pauli_channel(px, py, pz, qubit)` | `px + py + pz <= 1`; one channel per qubit | non-unitary; see below |
| `.depolarize2(p, pairs)` | `noise.depolarize2(p, q0, q1)` | one channel per `(q0, q1)` pair; indices in a pair must differ | non-unitary; see below |

Every noise channel is non-unitary: `circuit.adjoint()` raises `ValueError` naming it if the circuit contains one.

Coefficient rescale per channel: `depolarize(p)` scales every non-identity Pauli on its qubit by `1 - 4p/3`; `dephase(p)` scales `X` and `Y` by `1 - 2p` and leaves `Z` untouched; `pauli_channel(px, py, pz)` scales `X` by `1 - 2(py + pz)`, `Y` by `1 - 2(px + pz)`, `Z` by `1 - 2(px + py)`; `depolarize2(p)` scales every non-identity two-qubit Pauli on its pair by `1 - 16p/15`.

`pauli_channel(p/3, p/3, p/3, q)` is `depolarize(p, q)`; `pauli_channel(0, 0, p, q)` is `dephase(p, q)`.
See [Noise channels](../manual/circuits.md#noise-channels) for a worked example.

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
| `paulistrings.interop.load_task(path)` | a whole task-JSON schema-v1 file (or an already-parsed dict), returned as a `Task` dataclass with `.circuit`, `.observable`, `.truncation`, `.direction`, `.threads`, `.state` and the raw dict |

See [Importing circuits](../manual/circuits.md#importing-circuits) for the recipe.
