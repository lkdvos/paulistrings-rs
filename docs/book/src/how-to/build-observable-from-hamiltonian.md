# Observable from a Hamiltonian

A Hamiltonian is just a weighted sum of Pauli strings, so it is built the same way as any other observable — accumulate `{pauli_string: coefficient}` and hand it to `PauliSum.from_strings`.
This builds a transverse-field Ising chain, `H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)`:

```python
from paulistrings import PauliSum

n = 4
J, h = 1.0, 0.5
terms: dict[str, float] = {}

for i in range(n - 1):
    key = ["I"] * n
    key[i] = key[i + 1] = "Z"
    terms["".join(key)] = terms.get("".join(key), 0.0) - J

for i in range(n):
    key = ["I"] * n
    key[i] = "X"
    terms["".join(key)] = terms.get("".join(key), 0.0) - h

hamiltonian = PauliSum.from_strings(terms, num_qubits=n)
```

`terms` is a plain `dict`, so accumulating into `terms.get(key, 0.0) - J` before the call is what merges repeated bonds onto one string — `from_strings` itself never sees a duplicate key, since a Python dict cannot hold one.
For thousands of terms built programmatically, [`from_arrays`](build-observable-from-arrays.md) avoids the per-term string allocation and instead sums duplicate rows itself.

See [PauliSum reference](../reference/pauli-sum.md) for the full constructor surface and [Observable from Pauli strings](build-observable-from-strings.md) for the string-key convention.
