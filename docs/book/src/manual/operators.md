# Operators

## What a Pauli sum is

A Pauli sum is an operator on `n` qubits written in the Pauli basis: `O = Σ_P c_P P`, one complex coefficient per Pauli string `P ∈ {I, X, Y, Z}^n`, each string appearing at most once.
It is the one storage type in this library: the observable you start from, the Hamiltonian you Trotterize, the result a propagation hands back, and the "state" an overlap is taken against are all the same `PauliSum`.
The convention is Hermitian everywhere a Pauli string is written or read: a coefficient multiplies the literal Hermitian Pauli string, and `Y` carries no phase of its own.
That is the same convention stim uses, and it differs from the phased "canonical" `Y` some operator libraries carry, so a coefficient that reads `+1` here reads `+1` in stim and may not in a library that stores `iXZ`.

What a sum costs is its term count, not its qubit count; 127 qubits with a thousand terms is cheap and 4 qubits with a million terms is not, which is why every accessor below reports terms and every result page reports how many survived.
[Cost is terms, not qubits](propagation/index.md#cost-model) is the full statement.

## From Pauli strings {#from-pauli-strings}

The direct way to write an observable is a dict from Pauli strings to coefficients.
Each key is exactly `num_qubits` characters of `I`, `X`, `Y`, `Z` (upper case only), and character `i` addresses qubit `i`.

```python
from paulistrings import PauliSum

observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)
print(len(observable), observable.num_qubits)
```

```text
4 4
```

A wrong key length or a character outside `IXYZ` is a `ValueError` naming the offending string; an exact-zero coefficient is dropped rather than stored.
A Python dict cannot hold a duplicate key, so merging repeated strings is your job before the call, which is what the next section does.
The constructor table is in [PauliSum](../library/pauli-sum.md#constructors); [First propagation](../examples/first-propagation.md) runs this exact observable through a circuit.

## Hamiltonians and programmatic construction {#hamiltonians}

A Hamiltonian is a weighted sum of Pauli strings and is built the same way, by accumulating a dict and handing it over once.
This is a transverse-field Ising chain, `H = -J Σ Z_i Z_{i+1} - h Σ X_i`:

```python
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
print(len(hamiltonian))
```

```text
7
```

The `terms.get(key, 0.0) - J` accumulation is what merges a bond that appears twice onto one string; `from_strings` never sees a duplicate.
For tens of thousands of terms built in a loop, the string keys become the cost, and [`from_arrays`](#symplectic-arrays) below takes the bit columns directly and sums duplicate rows itself.

Keep two roles apart.
A Hamiltonian as a `PauliSum` is an *observable*, something you measure the energy of.
To evolve *under* a Hamiltonian, it enters as a circuit of Pauli rotations, one per term per Trotter step, with the angle rule in [Building a circuit](circuits.md#building-a-circuit); the observable that propagates is a different sum.
[Incremental propagation](propagation/incremental.md) is the Trotter time series end to end, and the shared constructions the example suite uses live in [`examples/common/observables.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/common/observables.py).

## Symplectic arrays {#symplectic-arrays}

Underneath, a Pauli string is two bit vectors: qubit `q`'s Pauli is bit `q` of an `x` word and bit `q` of a `z` word, with `(0,0)` = `I`, `(1,0)` = `X`, `(0,1)` = `Z` and `(1,1)` = `Y`.
Words are 64 bits, and a sum's `width` is how many words each term carries: one for up to 64 qubits, two for up to 128, then 4, 8 and 16, so 1024 qubits is the ceiling.
`x_array()` and `z_array()` return the two columns as `uint64` arrays of shape `(len, width)`, where column `j` covers qubits `64*j .. 64*j + 63`; `coefficients_array()` is the matching `complex128` column.

All three are snapshots — owned copies, not views — so writing into one leaves the sum untouched, and the three taken together are the sum at one instant.
`from_arrays` is the inverse, taking the same three columns back into a `PauliSum`.
Duplicate rows sum their coefficients, arrays narrower than the sum's width tier are zero-padded, and a set bit at or beyond `num_qubits` is a `ValueError`.

```python
import numpy as np

x = hamiltonian.x_array()
z = hamiltonian.z_array()
c = hamiltonian.coefficients_array()
print(x.dtype, x.shape, c.dtype, hamiltonian.width)

rebuilt = PauliSum.from_arrays(x, z, c, num_qubits=n)
print(len(rebuilt), np.allclose(rebuilt.coefficients_array(), c))
```

```text
uint64 (7, 1) complex128 1
7 True
```

There is no term-listing accessor; decoding is a bit loop over the words, and this one works at any width:

```python
PAULI = np.array([["I", "Z"], ["X", "Y"]])

def decode(x_row, z_row, num_qubits):
    return "".join(
        PAULI[(int(x_row[q // 64]) >> (q % 64)) & 1, (int(z_row[q // 64]) >> (q % 64)) & 1]
        for q in range(num_qubits)
    )

for row in np.argsort(-np.abs(c)):
    print(decode(x[row], z[row], n), f"{c[row].real:+.3f}")
```

```text
ZZII -1.000
IZZI -1.000
IIZZ -1.000
XIII -0.500
IXII -0.500
IIXI -0.500
IIIX -0.500
```

The rows come out in the sum's canonical storage order — [bucket](propagation/engine.md#layout) index ascending, then lexicographic `(x, z)` — which is the same for all three accessors called on the same sum but is neither insertion order nor string order, and a `propagate` call may rebucket and reorder.
Pair rows from one snapshot, never `x` from before a propagation with `c` from after it.
Layout and dtypes are tabulated under [PauliSum](../library/pauli-sum.md#accessors); [B1](../examples/showcases/b1-operator-scrambling.md) reads a light cone and an OTOC off these three arrays and [B6](../examples/showcases/b6-resource-probes.md) computes Pauli-spectrum entropies from them in pure NumPy.

## Inspecting a sum {#inspecting}

`len(sum)` is the term count, `.num_qubits` the register size, `.width` the word tier, and `PauliSum(n)` is the empty sum on `n` qubits, which is what a propagation that truncated everything returns.
`.coefficients()` is the coefficient column as a Python list, for small sums where NumPy is overkill.

The one scalar worth knowing by name is the Hilbert–Schmidt norm `Σ|c|²`, which equals `tr(O†O) / 2^n` because Pauli strings are orthonormal under that inner product; `sum.overlap(sum).real` computes it in one call.
Unitary evolution conserves it exactly, so after a truncated propagation the drop from the input's value to the output's is precisely the norm truncation deleted, which makes it the first diagnostic of any result.

```python
print(len(PauliSum(4)), PauliSum(4).width)
norm = float(np.sum(np.abs(hamiltonian.coefficients_array()) ** 2))
print(norm, hamiltonian.overlap(hamiltonian).real)
```

```text
0 1
4.0 4.0
```

Reading the retained norm alongside a cutoff sweep is [Validating a result](propagation/validation.md); [Overlaps and the identity coefficient](measurements.md#overlap) covers `overlap` between two different sums.

## Saving and loading {#saving-and-loading}

A sum crosses a process boundary as a `.npz` file: the three columns plus `num_qubits` and a format tag, written with `numpy.savez_compressed`, no pickle.

```python
from paulistrings import io as psio

psio.save("hamiltonian.npz", hamiltonian)
reloaded = psio.load("hamiltonian.npz")
print(len(reloaded), reloaded.num_qubits, psio.FORMAT)
```

```text
7 4 paulistrings-npz-v1
```

`save` appends `.npz` if the path lacks it, and `load` raises `ValueError` on a file without the `paulistrings-npz-v1` format tag rather than guessing at an archive's layout.
This is how a propagated observable is handed to a second run: propagate, save, load elsewhere, propagate further, which is the whole mechanism of [B5](../examples/showcases/b5-operator-backpropagation.md#validation).
The format is specified under [PauliSum](../library/pauli-sum.md#saving-loading-importing).

## Where else observables come from

Two importers hand back a `PauliSum` you did not write by hand.
`interop.circuit_from_stim` returns `(circuit, observable)`, where the observable is built from the stim program's `OBSERVABLE_INCLUDE` instructions, or `None` when it has none; `interop.load_task` parses a task-JSON file and its `.observable` is the task's observable when one is defined.
Both live in [Importing circuits](circuits.md#importing-circuits), since the circuit is the main thing they import.
The third source is a previous propagation: its result is an ordinary `PauliSum`, ready to be propagated further, which is what [Incremental propagation](propagation/incremental.md) is built on.

## See it in use

- [First propagation](../examples/first-propagation.md) — `from_strings`, the decoded surviving terms, and the retained norm across a cutoff sweep, on four qubits.
- [B1 — Operator scrambling](../examples/showcases/b1-operator-scrambling.md#running-it) — the light cone, OTOC and two-point function all read from `x_array`, `z_array` and `coefficients_array` exported once per Trotter step.
- [B6 — Resource probes](../examples/showcases/b6-resource-probes.md) — Pauli-spectrum entropy and operator entanglement computed in pure NumPy over the same three arrays.
- [B5 — Hybrid depth reduction](../examples/showcases/b5-operator-backpropagation.md#validation) — an evolved observable saved to `.npz`, read back, and embedded in a task JSON, with the round-trip gap measured.
