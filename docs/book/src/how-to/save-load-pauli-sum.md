# PauliSum save and load

```python
from paulistrings import PauliSum, io as psio

evolved = PauliSum.from_strings({"XI": 1.0, "IZ": 0.5}, num_qubits=2)
psio.save("evolved.npz", evolved)
reloaded = psio.load("evolved.npz")
```

This is the `paulistrings-npz-v1` format: an `np.savez_compressed` archive of the symplectic `x`/`z` columns and the coefficients, no pickle and no serde.
It's how an evolved observable crosses a process boundary — propagate, save, load elsewhere, propagate further, as in [Case study B5](../explanation/case-studies/b5-operator-backpropagation.md).
`load` hard-errors on a missing or unrecognized format field rather than guessing.
