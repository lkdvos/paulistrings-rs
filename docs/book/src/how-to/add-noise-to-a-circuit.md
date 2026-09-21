# Add noise to a circuit

```python
from paulistrings import Circuit

circuit = Circuit(4)
circuit.h(0)
circuit.depolarize(0.01, [0])          # one channel per qubit in the list
circuit.cnot(0, 1)
circuit.depolarize2(0.01, [(0, 1)])
circuit.dephase(0.01, [1])
circuit.amplitude_damping(0.01, [2])
circuit.pauli_channel(0.002, 0.002, 0.008, [3])
```

Each `Circuit` method above pushes one noise channel per qubit (or pair, for `depolarize2`) — a gate is a truncation point and so is a noise channel, so keep noise one channel per push rather than fusing it into the preceding gate.
The `noise` module exposes the same channels as free factory functions, for building a channel list before you have a `Circuit` to push onto, or reusing one across circuits:

```python
from paulistrings import noise

channel = noise.depolarize(0.01, qubit=0)
circuit.append(channel)
```

See [Circuit reference](../reference/circuit.md) for the full channel list and parameter meanings.
