# Operators

The primary objects of this library are operators acting on $L$ qubits.
Any such operator can be expanded in the Pauli basis as

$$
O = \sum_{i=1}^N c_i P_i,
$$

where every $P_i$ is a Pauli string, a string of $L$ letters from $\{I, X, Y, Z\}$, and every $c_i$ is a complex coefficient.
A generic operator has $4^L$ coefficients, but many operators of interest are sparse, and only the $N \ll 4^L$ nonzero terms are stored.

## Pauli strings

A single Pauli string is stored in the symplectic encoding.
For each qubit we store two bits $(x, z) \in \mathbb{F}_2^2$, which record the presence of an $X$ and a $Z$ factor, in a Hermitian convention:

$$
P(x, z) = i^{xz} X^x Z^z.
$$

In particular, $P(1, 0) = X$, $P(0, 1) = Z$ and $P(1, 1) = iXZ = Y$, since $XZ = -iY$.

A string on $L$ qubits is stored in $2L$ bits, collected in the symplectic vector $\boldsymbol{p} = (\boldsymbol{p}_x \,\vert\, \boldsymbol{p}_z) \in \mathbb{F}_2^{2L}$ with $x$- and $z$-parts $\boldsymbol{p}_x, \boldsymbol{p}_z \in \mathbb{F}_2^L$, and

$$
P(\boldsymbol{p}) = i^{N_Y(\boldsymbol{p})} \bigotimes_{j=0}^{L-1} X^{p_{x,j}} Z^{p_{z,j}},
\qquad
N_Y(\boldsymbol{p}) = |\boldsymbol{p}_x \wedge \boldsymbol{p}_z|.
$$

Here $\wedge$ is the bitwise AND and $|\boldsymbol{v}|$ is the Hamming weight of a bitvector $\boldsymbol{v}$, the number of its nonzero bits, so $N_Y(\boldsymbol{p})$ counts the $Y$ factors of the string.
Pauli strings are denoted by uppercase letters $P, Q$ and their symplectic vectors by the corresponding bold lowercase letters $\boldsymbol{p}, \boldsymbol{q}$.
All arithmetic on bits and bitvectors is over $\mathbb{F}_2$, with $\oplus$ denoting addition (XOR).
The weight of a string is the number of sites on which it acts nontrivially,

$$
w(\boldsymbol{p}) = |\boldsymbol{p}_x \vee \boldsymbol{p}_z|,
$$

with $\vee$ the bitwise OR.
Sites are numbered from $0$: site $j$ is character $j$ of a label and qubit index $j$ of every method.

### The `PauliString` type

`PauliString` is a single Pauli string, with no coefficient attached.
It is immutable and carries `num_qubits`, its `label`, and its `weight`, the quantity [`truncation.weight`](propagation/truncation.md#choosing-a-policy) caps.

```python
from paulistrings import PauliString, p

print(PauliString.identity(3), PauliString.x(0, 3), PauliString.y(1, 3), PauliString.z(2, 3))
print(p("XYZ"), p("XYZ").weight)
```

```text
III XII IYI IIZ
XYZ 3
```

`p(label)` is short for `PauliString.from_label(label)`.
Two strings compare equal when their labels agree.
The constructors and accessors are listed in [PauliString](../library/pauli-string.md).

### Multiplication

Multiplying two strings is an XOR of their symplectic vectors together with a phase:

$$
P(\boldsymbol{p})\, P(\boldsymbol{q}) = i^{k(\boldsymbol{p}, \boldsymbol{q})}\, P(\boldsymbol{p} \oplus \boldsymbol{q}),
$$

with

$$
k(\boldsymbol{p}, \boldsymbol{q}) = N_Y(\boldsymbol{p}) + N_Y(\boldsymbol{q}) - N_Y(\boldsymbol{p} \oplus \boldsymbol{q}) + 2\, |\boldsymbol{p}_z \wedge \boldsymbol{q}_x|,
$$

computed modulo 4.
`mul` returns the phase and the product string:

```python
phase, product = p("XZY").mul(p("ZXY"))
print(phase, product)
```

```text
(1+0j) YYI
```

It is a good exercise for anyone new to Pauli strings to check this output by hand.
The operator `*` between two strings raises a `TypeError`: `*` scales by a number, and `mul` is the product.

### Commutation

Two strings either commute or anticommute,

$$
P(\boldsymbol{p})\, P(\boldsymbol{q}) = (-1)^{s(\boldsymbol{p}, \boldsymbol{q})}\, P(\boldsymbol{q})\, P(\boldsymbol{p}),
$$

with the bilinear form

$$
s(\boldsymbol{p}, \boldsymbol{q}) = \boldsymbol{p}_x \cdot \boldsymbol{q}_z \oplus \boldsymbol{q}_x \cdot \boldsymbol{p}_z,
$$

where the dot products are taken over $\mathbb{F}_2$.
They commute when $s = 0$ and anticommute when $s = 1$.
In matrix form, $s(\boldsymbol{p}, \boldsymbol{q}) = \boldsymbol{p}^T J \boldsymbol{q}$ with $J = \begin{pmatrix} 0 & \mathbb{I}_L \\ \mathbb{I}_L & 0 \end{pmatrix}$, the symplectic form that gives the encoding its name.
The commutator and anticommutator are each either $2PQ$ or zero:

$$
[P, Q] = \big(1 - (-1)^{s(\boldsymbol{p}, \boldsymbol{q})}\big)\, PQ,
\qquad
\{P, Q\} = \big(1 + (-1)^{s(\boldsymbol{p}, \boldsymbol{q})}\big)\, PQ.
$$

`commutes_with` returns `True` if and only if $s(\boldsymbol{p}, \boldsymbol{q}) = 0$; `commutator` and `anticommutator` return the coefficient together with the string $PQ$:

```python
x, z = p("X"), p("Z")
print(x.commutes_with(z), x.commutator(z), x.anticommutator(z))
```

```text
False (-2j, Y) (0j, Y)
```

Here $[X, Z] = 2XZ = -2iY$ and $\{X, Z\} = 0$.
The string is reported in both cases, so a zero coefficient is an exact algebraic zero.

### Orthonormality

Pauli strings are orthonormal in the normalized Hilbert–Schmidt inner product,

$$
\langle P(\boldsymbol{p}), P(\boldsymbol{q}) \rangle = 2^{-L} \operatorname{Tr}\big(P(\boldsymbol{p})^\dagger P(\boldsymbol{q})\big) = \delta_{\boldsymbol{p}, \boldsymbol{q}}.
$$

The expansion $O = \sum_i c_i P_i$ is therefore unique, with coefficients

$$
c_i = \langle P_i, O \rangle = 2^{-L} \operatorname{Tr}(P_i\, O),
$$

and $O$ is Hermitian if and only if every $c_i$ is real.
For $O_1 = \sum_i a_i P(\boldsymbol{p}_i)$ and $O_2 = \sum_j b_j P(\boldsymbol{q}_j)$, only strings present in both operators contribute:

$$
\langle O_1, O_2 \rangle = \sum_{i, j \colon \boldsymbol{p}_i = \boldsymbol{q}_j} \bar{a}_i b_j,
\qquad
\lVert O \rVert_2^2 = \langle O, O \rangle = \sum_i |c_i|^2.
$$

`overlap` computes $\langle O_1, O_2 \rangle$:

```python
a, b = 1.0 * p("XY"), 1.0 * p("YX")
print(a.overlap(a), a.overlap(b))
```

```text
(1+0j) 0j
```

Unitary conjugation preserves $\lVert O \rVert_2$; one can use this to [validate results](propagation/validation.md), i.e. measure what truncation removed.

## Pauli sums

### Storage and construction {#the-paulisum-type}

The `PauliSum` type stores the expansion as three columns, one row per term: the $x$-parts $\boldsymbol{p}_x$, the $z$-parts $\boldsymbol{p}_z$, and the coefficients $c_i$.
No string appears twice.
Combining two sums therefore requires the union of their strings with the coefficients of equal strings added, a reduce-by-key operation called the *deduplication step*, after which terms whose coefficient is exactly zero are dropped.
Every constructor and every addition of sums ends with this step.

A sum can be written as on paper:

```python
from paulistrings import PauliSum, p

observable = 0.25 * p("XIII") + 0.25 * p("IXII") + 0.25 * p("IIXI") + 0.25 * p("IIIX")
print(observable)
print(PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}))
```

```text
0.25*XIII + 0.25*IXII + 0.25*IIXI + 0.25*IIIX
0.25*XIII + 0.25*IXII + 0.25*IIXI + 0.25*IIIX
```

A string times a number is a one-term sum, and `+` adds sums.
Each `+` builds a new sum, so assembling $N$ terms one at a time costs $O(N^2)$.
`PauliSum.from_strings` takes all terms at once, as a dict from label to coefficient or as two equal-length sequences of labels and coefficients, in which a repeated label accumulates.

### Arithmetic {#combining-sums}

`+` and `-` combine two sums with the deduplication step, `+=` and `-=` do the same in place, and `*` scales every coefficient by a number:

```python
a = PauliSum.from_strings({"ZZII": -1.0})
b = PauliSum.from_strings({"IZZI": -1.0})
print(a + a)
print(2.0 * (a + b))
```

```text
-2*ZZII
-2*ZZII + -2*IZZI
```

Both operands must have the same `num_qubits`.
The product of two sums is not implemented, and `*` between two `PauliSum` objects raises `TypeError`.

### Sparse terms {#sparse-terms}

Writing every label in full is tedious when $L$ is large.
`pt(label, sites)` names only the sites a string acts on, `label[k]` on `sites[k]`, and returns a `PauliTerms`: an ordered list of terms with no qubit count.
`PauliSum(L, terms)` binds the terms to $L$ qubits.

```python
from paulistrings import pt

terms = -1.0 * pt("ZZ", [(0, 1), (1, 2)]) - 0.5 * pt("X", range(3))
print(len(terms), terms)
print(PauliSum(3, terms))
print(PauliSum(2, pt("X", 0) + pt("X", 0)))
```

```text
5 -1*Z0Z1 + -1*Z1Z2 + -0.5*X0 + -0.5*X1 + ... (1 more term)
-1*ZZI + -1*IZZ + -0.5*XII + -0.5*IXI + ... (1 more term)
2*XI
```

`sites` is a single site (`pt("X", 3)`), one tuple (`pt("ZZ", (0, 1))`), or a list of tuples (`pt("ZZ", bonds)`), giving one term per tuple.
With a one-letter label, a list of sites gives one term per site, as in `pt("X", range(3))`.
A `PauliTerms` supports `+`, `-`, scaling by a number and `sum()`, and keeps the terms in the order they were written, repeats included.
The repeats are merged only when the terms are bound, as in the last line.
The order matters when the terms define a Trotter circuit ([Trotter steps](circuits.md#trotter-steps)).

### A Hamiltonian

The transverse-field Ising chain with open boundaries,

$$
H = -J \sum_{j=0}^{L-2} Z_j Z_{j+1} - h \sum_{j=0}^{L-1} X_j,
$$

is written with `pt` and the bond list of the chain:

```python
from paulistrings.lattices import chain

L, J, h = 4, 1.0, 0.5
H = -J * pt("ZZ", chain(L)) - h * pt("X", range(L))
hamiltonian = PauliSum(L, H)
print(hamiltonian)
```

```text
-1*ZZII + -1*IZZI + -1*IIZZ + -0.5*XIII + ... (3 more terms)
```

`lattices.chain(L)` and `lattices.square(lx, ly)` return bond lists, with `periodic=True` for periodic boundaries.
`models.transverse_ising(bonds, sites, J, h)` and `models.xxz(bonds, J, Jz)` build the corresponding `PauliTerms` directly.

The same $H$ generates time evolution.
For $H = \sum_k h_k Q_k$, each Trotter factor is a Pauli rotation,

$$
\exp(-i h_k\, \delta t\, Q_k) = R_{Q_k}(2 h_k\, \delta t),
\qquad
R_Q(\theta) = \exp(-i \theta Q / 2),
$$

and `trotter(H, dt, L)` builds these rotations from the `PauliTerms`, so the angles never need to be computed by hand ([Trotter steps](circuits.md#trotter-steps)).

### Symplectic arrays {#symplectic-arrays}

`x_array()` and `z_array()` return the $x$- and $z$-parts of all $N$ terms as `uint64` arrays of shape `(N, width)`, with site $j$ in bit `j % 64` of column `j // 64`.
`coefficients_array()` returns the $c_i$ as a `complex128` array of length $N$.
All three are copies, and all three list the terms in the same canonical order: partition-bucket index, then lexicographic order of $(\boldsymbol{p}_x, \boldsymbol{p}_z)$.
This order is neither insertion order nor label order, and `propagate` can change it, so rows are paired only within one snapshot.
`PauliSum.from_arrays` is the inverse: duplicate rows are summed, arrays narrower than the sum's width are zero-padded, and a bit set at or beyond site $L$ raises `ValueError`.

```python
import numpy as np

x, z, c = hamiltonian.x_array(), hamiltonian.z_array(), hamiltonian.coefficients_array()
print(x.dtype, x.shape, c.dtype, c.shape)

rebuilt = PauliSum.from_arrays(x, z, c, num_qubits=L)
print(len(rebuilt), np.allclose(rebuilt.coefficients_array(), c))
```

```text
uint64 (7, 1) complex128 (7,)
7 True
```

A label is recovered from the two bits of each site, using the encoding above:

```python
LETTER = np.array([["I", "Z"], ["X", "Y"]])  # LETTER[x bit, z bit]

def bit(words, j):
    return (int(words[j // 64]) >> (j % 64)) & 1

for row in range(len(c)):
    letters = "".join(LETTER[bit(x[row], j), bit(z[row], j)] for j in range(L))
    print(letters, c[row].real)
```

```text
ZZII -1.0
IZZI -1.0
IIZZ -1.0
XIII -0.5
IXII -0.5
IIXI -0.5
IIIX -0.5
```

The three $ZZ$ strings come first because their $x$-parts are zero.

### Inspecting a sum {#inspecting}

`len(...)` is the number of terms $N$, `num_qubits` is $L$, and `width` is the number of words per part.
`overlap` is the inner product $\langle O_1, O_2 \rangle$, so `O.overlap(O)` is $\lVert O \rVert_2^2$:

```python
print(len(hamiltonian), hamiltonian.num_qubits, hamiltonian.width)
print(hamiltonian.overlap(hamiltonian).real, np.sum(np.abs(c) ** 2))
```

```text
7 4 1
4.0 4.0
```

`PauliSum(L)` is the empty sum on $L$ qubits, the zero operator, and prints as `0`.

## Saving and loading {#saving-and-loading}

A sum is saved as a `.npz` archive holding the three columns, `num_qubits`, and a format tag:

```python
from paulistrings import io as psio

psio.save("hamiltonian.npz", hamiltonian)
reloaded = psio.load("hamiltonian.npz")
print(len(reloaded), reloaded.num_qubits, psio.FORMAT)
```

```text
7 4 paulistrings-npz-v1
```

`save` appends `.npz` when the path lacks it, and `load` raises `ValueError` on an archive without the `paulistrings-npz-v1` tag.