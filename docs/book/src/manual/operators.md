# Operators

The primary objects of this library are operators acting on $L$ qubits.
For a single qubit, the identity along with the Pauli matrices forms an orthonormal basis.
Therefore, we can generate any multi-qubit operator as a linear combination of so-called **Pauli strings**.
These are strings of length $L$ containing the letters $\{I, X, Y, Z\}$, such that all possible strings carry all possible basis elements.

## Pauli strings

A single Pauli string $P$ is stored in the so-called symplectic encoding.
This means that for each qubit, we store two separate bits that dictate what operator is used.
Concretely, for bits $x$ and $z$, we encode the operator in a Hermitian convention as:

$$
P = j^{xz}X^xZ^z
$$

In particular, this gives $X = (1, 0)$, $Z = (0, 1)$, and $Y = jXZ = j(1, 1)$, equivalently $XZ = -jY$.

For an $L$-qubit string, we can collect these into two masks, or in components:

$$
P(\vec{x}, \vec{z}) = j^{\sum_q^L x_qz_q} \bigotimes_q^L X_q^{x_q} Z_q^{z_q}
$$


### The `PauliString` type

A single Pauli string is its own type, `PauliString` — the object the symplectic encoding above describes, before any coefficient is attached to it.
It is immutable and carries `num_qubits`, a `weight` (the number of non-identity factors, the same quantity [`truncation.weight`](propagation/truncation.md#choosing-a-policy) caps), and a `label` (the `IXYZ` string).
`str(p)` and `repr(p)` both print the label directly — a bare Pauli string has nothing else worth showing — so it pretty-prints itself in a REPL or a plain `print()` call with no extra step.

```python
from paulistrings import PauliString, p

identity = PauliString.identity(3)
x0 = PauliString.x(0, 3)
y1 = PauliString.y(1, 3)
z2 = PauliString.z(2, 3)

print(identity, identity.weight)
print(x0, y1, z2)
print(p("XYZ") == x0.mul(y1)[1].mul(z2)[1])
```

```text
III 0
XII IYI IIZ
True
```

`identity`, `x`, `y`, `z` and `from_label` are the five constructors on the class itself; `num_qubits` is a plain positional argument, not a required keyword, and each single-site constructor takes the qubit index first.
`p(label)` is a package-level shorthand for `PauliString.from_label(label)`, for the common case of writing one down by hand.
Two strings compare equal when their labels and `num_qubits` agree; there is no coefficient here to compare, which is exactly the difference from a one-term `PauliSum`.
The constructor, accessor and operation tables are in [PauliString](../library/pauli-string.md).

### Single string operations

Four operations on bare Pauli strings are closed-form from the symplectic encoding: whether two strings (anti-)commute, their product, and their (anti-)commutator.

`mul` is the closed form itself: multiplying two Pauli strings results in exactly one output string, up to a phase $j^k$ the caller must keep track of.
The canonical example being $XZ = -jY$ above.
In general, writing $(x^P, z^P)$ and $(x^Q, z^Q)$ for the two operands' symplectic bits, the phase is:

$$
j^k, \qquad k = \sum_q \Big(x^P_q z^P_q + x^Q_q z^Q_q - (x^P_q \oplus x^Q_q)(z^P_q \oplus z^Q_q)\Big) + 2 \sum_q z^P_q x^Q_q \pmod 4
$$

```python
from paulistrings import PauliString

a, b = PauliString.from_label("XZY"), PauliString.from_label("ZXY")
phase, product = a.mul(b)
print(phase, product)
```

```text
(1+0j) YYI
```

`commutes_with` is the symplectic inner product read as a boolean; `anticommutes_with` is its negation, since a pair of Pauli strings always does one or the other and never neither.
Generally, the computation follows:

$$
\langle P, Q \rangle = \sum_q \Big(x^P_q z^Q_q + z^P_q x^Q_q\Big) \bmod 2
$$

with the two strings commuting exactly when $\langle P, Q \rangle = 0$:

```python
a, b = PauliString.from_label("XY"), PauliString.from_label("XY")
print(a.commutes_with(b), a.anticommutes_with(b))
```

```text
True False
```

Finally, both `commutator` and `anticommutator` result in a single string and coefficient again.
Generally these simplify and again a closed-form formula exists, reusing $\langle P, Q \rangle$ and `mul`'s phase from above:

$$
[P, Q] = \begin{cases} 2 \cdot \text{mul}(P, Q) & \langle P, Q \rangle = 1 \\ 0 & \langle P, Q \rangle = 0 \end{cases}
\qquad
\{P, Q\} = \begin{cases} 2 \cdot \text{mul}(P, Q) & \langle P, Q \rangle = 0 \\ 0 & \langle P, Q \rangle = 1 \end{cases}
$$

so an exact zero is a `0` coefficient here, never a `PauliSum` with nothing worth holding it:

```python
x, z = PauliString.x(0, 1), PauliString.z(0, 1)
print(x.commutator(z), x.anticommutator(z))
```

```text
(-2j, Y) (0j, Y)
```

That closure — one string in, one string and a phase out — is why a gate's image is a short, enumerable list of terms rather than a combinatorial explosion; [Cost is terms, not qubits](propagation/index.md#cost-model) is what a whole sum inherits from this one-string fact.

## Pauli sums

As the basis of Pauli strings spans the full operator space, any operator can be written as a linear combination of Pauli strings: $O = \sum_P c_P P$.
Since the total space is $4^L$-dimensional, we wish to store a **sparse** representation of the unique non-zero coefficient-string pairs.
To allow for efficient merging of the strings, we store two separate lists, one for the coefficients and one for the strings, sorted by the string.

### The `PauliSum` type {#the-paulisum-type}

A `PauliSum` is an operator on $L$ qubits written in the Pauli basis, $O = \sum_P c_P P$, one (complex) coefficient per `PauliString` $P$, each string appearing at most once.
It is the one storage type in this library: the observable you start from, the Hamiltonian you Trotterize, the result a propagation hands back, and the "state" an overlap is taken against are all the same `PauliSum`.
The convention is Hermitian everywhere a Pauli string is written or read: a coefficient multiplies the literal Hermitian Pauli string, and $Y$ carries no phase of its own.

What a sum costs is its term count $N$, not its qubit count $L$; $L = 127$ with $N$ in the thousands is cheap and $L = 12$ with $N$ in the millions is not, which is why every accessor below reports terms and every result page reports how many survived.
[Cost is terms, not qubits](propagation/index.md#cost-model) is the full statement.

The natural way to build a sum is out of single strings, combined with `+` and `*` the same way you would write the sum on paper: `p(label)` gives a single `PauliString`, and multiplying one by a number gives a one-term `PauliSum` that `+` then combines with the rest.

```python
from paulistrings import PauliSum, p

observable = 0.25 * p("XIII") + 0.25 * p("IXII") + 0.25 * p("IIXI") + 0.25 * p("IIIX")
print(observable)
```

```text
0.25*XIII + 0.25*IXII + 0.25*IIXI + 0.25*IIIX
```

Each label is exactly `num_qubits` characters of `I`, `X`, `Y`, `Z` (upper case only), character `i` addressing qubit `i`.
A wrong length or a character outside `IXYZ` is a `ValueError` naming the offending string, and multiplying a string by an exact-zero coefficient gives the empty sum rather than a stored zero.
`PauliString * PauliString` is a `TypeError`, not the Pauli product — that is `.mul(other)`, covered under [Single string operations](#single-string-operations) below.

`from_strings` builds a sum with many terms at once, straight from a full dict or from two parallel sequences of labels and coefficients, inferring `num_qubits` from the first label when it isn't given:

```python
observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}
)
print(observable)
```

```text
0.25*XIII + 0.25*IXII + 0.25*IIXI + 0.25*IIIX
```

The constructor table is in [PauliSum](../library/pauli-sum.md#constructors); [First propagation](../examples/first-propagation.md) runs this exact observable through a circuit.

**The Hermitian convention above is a storage detail, not an input or display rule.**
It does not change what `from_strings` accepts or what a decoded label reads back as — `Y` is always literally `Y`, on the way in and on the way out.
It matters only when comparing a coefficient against another library's internal representation: stim stores `Y` the same Hermitian way, so a coefficient that reads `+1` here reads `+1` in stim, but a library that keeps the phased "canonical" $Y = jXZ$ instead would read the same physical operator's coefficient differently.

### Multi-string operations

#### Combining sums {#combining-sums}

The same `+` and `*` above also combine two full sums, not just single strings: `+` and `-` add or subtract coefficients on matching strings and keep the rest, `+=` and `-=` do the same in place, and `*`/`*=` scale every coefficient by a number.

```python
from paulistrings import PauliSum

bond = PauliSum.from_strings({"ZZII": -1.0})
combined = bond + bond
combined += PauliSum.from_strings({"IZZI": -1.0})
combined *= 2.0

print(combined)
print(combined.overlap(bond).real)
```

```text
-4*ZZII + -2*IZZI
4.0
```

`bond + bond` adds onto the shared `ZZII` string instead of needing `2 * bond`'s coefficient computed by hand, and the `+=` after it adds a string that was not there yet without disturbing `ZZII`.
This does not extend to multiplying two sums together — that is a full operator product, a much larger and entirely different operation this page does not cover — so `*`/`*=` on a `PauliSum` is scalar-only.

#### Hamiltonians and programmatic construction

A Hamiltonian is a weighted sum of Pauli strings and is built the same way, by accumulating a dict and handing it over once.
This is a transverse-field Ising chain, $H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i$:

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
print(hamiltonian)
```

```text
-1*ZZII + -1*IZZI + -1*IIZZ + -0.5*XIII + ... (3 more terms)
```

The `terms.get(key, 0.0) - J` accumulation is what merges a bond that appears twice onto one string; `from_strings` never sees a duplicate.
For tens of thousands of terms built in a loop, the string keys become the cost, and [`from_arrays`](#symplectic-arrays) below takes the bit columns directly and sums duplicate rows itself.

Keep two roles apart.
A Hamiltonian as a `PauliSum` is an *observable*, something you measure the energy of.
To evolve *under* a Hamiltonian, it enters as a circuit of Pauli rotations, one per term per Trotter step, with the angle rule in [Building a circuit](circuits.md#building-a-circuit); the observable that propagates is a different sum.
[Incremental propagation](propagation/incremental.md) is the Trotter time series end to end, and the shared constructions the example suite uses live in [`examples/common/observables.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/common/observables.py).

#### Symplectic arrays {#symplectic-arrays}

Underneath, a Pauli string is two bit vectors: qubit `q`'s Pauli is bit `q` of an `x` word and bit `q` of a `z` word, with $(0,0) = I$, $(1,0) = X$, $(0,1) = Z$ and $(1,1) = Y$.
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

#### Inspecting a sum {#inspecting}

`len(...)` gives the term count $N$, `.num_qubits` the qubit count $L$, and `.width` the word tier; `PauliSum(4)` below constructs the empty sum on four qubits — printing it shows `0`, the zero operator, which is what a propagation that truncated everything returns.
`.coefficients()` is the coefficient column as a Python list, for small sums where NumPy is overkill.

The one scalar worth knowing by name is the Hilbert–Schmidt norm $\sum_P |c_P|^2$, which equals $\text{tr}(O^\dagger O) / 2^L$ because Pauli strings are orthonormal under that inner product; calling `.overlap()` on a sum against itself computes it in one call.
Unitary evolution conserves it exactly, so after a truncated propagation the drop from the input's value to the output's is precisely the norm truncation deleted, which makes it the first diagnostic of any result.

```python
print(PauliSum(4), PauliSum(4).width)
norm = float(np.sum(np.abs(hamiltonian.coefficients_array()) ** 2))
print(norm, hamiltonian.overlap(hamiltonian).real)
```

```text
0 1
4.0 4.0
```

Reading the retained norm alongside a cutoff sweep is [Validating a result](propagation/validation.md); [Overlaps and the identity coefficient](measurements.md#overlap) covers `overlap` between two different sums.

## IO

### Saving and loading {#saving-and-loading}

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

### Importing from external sources

Two importers hand back a `PauliSum` you did not write by hand.
`interop.circuit_from_stim` returns `(circuit, observable)`, where the observable is built from the stim program's `OBSERVABLE_INCLUDE` instructions, or `None` when it has none; `interop.load_task` parses a task-JSON file and its `.observable` is the task's observable when one is defined.
Both live in [Importing circuits](circuits.md#importing-circuits), since the circuit is the main thing they import.
The third source is a previous propagation: its result is an ordinary `PauliSum`, ready to be propagated further, which is what [Incremental propagation](propagation/incremental.md) is built on.

## References

- [First propagation](../examples/first-propagation.md) — `from_strings`, the decoded surviving terms, and the retained norm across a cutoff sweep, on four qubits.
- [B1 — Operator scrambling](../examples/showcases/b1-operator-scrambling.md#running-it) — the light cone, OTOC and two-point function all read from `x_array`, `z_array` and `coefficients_array` exported once per Trotter step.
- [B6 — Resource probes](../examples/showcases/b6-resource-probes.md) — Pauli-spectrum entropy and operator entanglement computed in pure NumPy over the same three arrays.
- [B5 — Hybrid depth reduction](../examples/showcases/b5-operator-backpropagation.md#validation) — an evolved observable saved to `.npz`, read back, and embedded in a task JSON, with the round-trip gap measured.
- [PauliString](../library/pauli-string.md) — the dry constructor, accessor and operation reference for a single string.
- [PauliSum](../library/pauli-sum.md) — the dry constructor, accessor and format reference this chapter cites throughout.
