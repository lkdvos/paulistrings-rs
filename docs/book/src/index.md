# paulistrings-rs

{{#include ../../../README.md:pitch}}

Inspired by [`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl).

## Quickstart

In many quantum simulations, the object of interest is typically the expectation value $\langle O \rangle = \text{tr}(\rho U^\dagger O U)$.
Here we start from some initial density matrix $\rho$ which is evolved through a circuit $U$ and then measured with an operator $O$.
For Pauli propagation methods, instead we work in the Heisenberg picture and work backwards: we start from the operator $O$, which is then evolved backwards through the circuit $U$, and finally measured against a density matrix $\rho$.

### Simple circuit

As a simple example, we may look at a four-qubit initial state $|++++\rangle$, and measure the total magnetization after propagating through the following simple circuit:

![Circuit diagram: Rz(pi/8) on qubit 0, then a CNOT from qubit 0 to qubit 1, then H on qubit 2; qubit 3 is idle](assets/quickstart/circuit.svg)

```python
import math
from paulistrings import Circuit, PauliSum

observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)

circuit = Circuit(4)
circuit.rz(math.pi / 8, 0)
circuit.cnot(0, 1)
circuit.h(2)

evolved = observable.propagate(circuit, direction="heisenberg")
print(len(evolved), evolved.expectation("x+").real)
```


```text
5 0.7309698831278217
```

### Quenched time-evolution

For a slightly more involved example, we can consider the 2D transverse-field Ising model, and measure the magnetization after a quench from the $|+\rangle^{\otimes N}$.

$$
H = -J \sum_{\langle i, j \rangle} Z_i Z_j - h \sum_i X_i
$$

This is achieved by propagating the total magnetization $M = \sum_i X_i$ through a Trotterized circuit and measuring against the initial state $|+\rangle\langle+|^{\otimes N}$.
However, in order to keep this computation tractable at longer times, we truncate the intermediate sums of strings, for example by keeping only the largest-magnitude terms up to a fixed count.
Finally, we can extrapolate to the limit of no truncation to validate our results.

![Average X magnetization vs time for a 6x6 periodic Ising quench, two field strengths h (color) and five TopN term caps each (opacity)](assets/ising-quench-convergence/quench_convergence.svg)

Two field strengths `h` (color) and five `TopN` term caps `10², 10³, 10⁴, 10⁵, 3×10⁵` (opacity, faintest at the smallest cap) on a 6×6 periodic lattice, `J = 1`.
At `h = 0.3` the three largest caps already sit on top of each other through `t = 0.8`: that overlap is the trusted regime, where raising the cap further would not move the curve.
At `h = 3.1` even the two largest caps are still visibly tightening rather than fully flat by `t = 0.8` — the same term budget buys less trustworthy time here, because a bigger single-qubit rotation per Trotter step spreads the operator across more of the Pauli sum faster.
The smallest cap, `TopN = 10²`, peels away from the trusted band by `t ≈ 0.2` at both field strengths, well before the run ends.
Figure script: `docs/figures/ising-quench-convergence/quench_convergence.py`.

```python
from paulistrings import truncation


def x_magnetization(lx, ly):
    n = lx * ly
    terms = {}
    for site in range(n):
        key = ["I"] * n
        key[site] = "X"
        terms["".join(key)] = 1.0 / n
    return PauliSum.from_strings(terms, num_qubits=n)


def trotter_step(lx, ly, dt, J=1.0, h=1.0):
    n = lx * ly
    circuit = Circuit(n)

    def idx(x, y):
        return (y % ly) * lx + (x % lx)

    for y in range(ly):
        for x in range(lx):
            i = idx(x, y)
            for nx, ny in ((x + 1, y), (x, y + 1)):
                j = idx(nx, ny)
                circuit.pauli_rotation("ZZ", [i, j], 2 * J * dt)
    for site in range(n):
        circuit.rx(2 * h * dt, site)
    return circuit


lx, ly = 3, 3
dt = 0.1
step_circuit = trotter_step(lx, ly, dt)
observable = x_magnetization(lx, ly)
for k in range(1, 6):
    observable = observable.propagate(step_circuit, truncation.coeff(1e-6), direction="heisenberg")
    print(f"t={k * dt:.2f}  <X>={observable.expectation('x+').real:+.6f}  terms={len(observable)}")
```

```text
t=0.10  <X>=+0.922619  terms=144
t=0.20  <X>=+0.720483  terms=7323
t=0.30  <X>=+0.472221  terms=23067
t=0.40  <X>=+0.271814  terms=57493
t=0.50  <X>=+0.183984  terms=98698
```

The magnetization falls and the tracked term count climbs by more than two orders of magnitude over five steps at fixed qubit count.

## Sections

| | |
|---|---|
| [Installation](installation.md) | install the Python package |
| [Manual](manual/index.md) | operators, circuits, engine and propagation, measurements — read start to finish, or by topic |
| [Examples](examples/index.md) | a guided first propagation, measured showcases and benchmarks, and comparisons against other tools |
| [Library](library/index.md) | this book's Python API reference |
| [Rust API](api/paulistrings/index.html) | rustdoc for the core crate, plus [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) on GitHub — the hand-off point for Rust users, not part of this book |


Source, issues and the full research record: [github.com/lkdvos/paulistrings-rs](https://github.com/lkdvos/paulistrings-rs).
Dual-licensed MIT OR Apache-2.0.
