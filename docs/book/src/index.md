# paulistrings-rs

{{#include ../../../README.md:pitch}}

Inspired by [`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl).

## Quickstart

Many quantum simulations ask for an expectation value $\langle O \rangle = \operatorname{tr}(\rho\, U^\dagger O U)$: a state $\rho$ evolved by a circuit $U$ and measured with an operator $O$.
Pauli propagation works in the Heisenberg picture: it evolves the operator $O$ through the circuit, from the last gate back to the first, and measures the result against $\rho$ at the end.

### Simple circuit

Start from $|{+}{+}{+}{+}\rangle$, apply the circuit below, and measure the average magnetization $\frac14 \sum_j X_j$:

![Circuit diagram: Rz(pi/8) on qubit 0, then a CNOT from qubit 0 to qubit 1, then H on qubit 2; qubit 3 is idle](assets/quickstart/circuit.svg)

```python
import math
from paulistrings import Circuit, PauliSum, pt

observable = PauliSum(4, pt("X", range(4)) / 4)

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

### Quenched time evolution

The 2D transverse-field Ising model on a periodic lattice,

$$
H = -J \sum_{\langle i, j \rangle} Z_i Z_j - h \sum_i X_i,
$$

is quenched from $|+\rangle^{\otimes N}$ by propagating the average magnetization $M = \frac{1}{N}\sum_i X_i$ through a Trotterized circuit.
To keep the sum tractable at longer times, the intermediate sums are truncated after every gate.

```python
from paulistrings import pt, trotter, truncation
from paulistrings.lattices import square

lx, ly, J, h, dt = 3, 3, 1.0, 1.0, 0.1
n = lx * ly
H = -J * pt("ZZ", square(lx, ly, periodic=True)) - h * pt("X", range(n))
step = trotter(H, dt, n)

observable = PauliSum(n, pt("X", range(n)) / n)
for k in range(1, 6):
    observable = observable.propagate(step, truncation.coeff(1e-6), direction="heisenberg")
    print(f"t={k * dt:.2f}  <X>={observable.expectation('x+').real:+.6f}  terms={len(observable)}")
```

```text
t=0.10  <X>=+0.922619  terms=144
t=0.20  <X>=+0.720483  terms=7323
t=0.30  <X>=+0.472221  terms=23067
t=0.40  <X>=+0.271814  terms=57493
t=0.50  <X>=+0.183984  terms=98698
```

The magnetization falls while the number of tracked terms grows by more than two orders of magnitude over five steps, at a fixed qubit count.

A truncated result is trusted only where raising the truncation budget no longer moves it.
The figure runs the same quench on a 6×6 lattice with `J = 1`, at two field strengths `h` (color) and five `TopN` term caps `10², 10³, 10⁴, 10⁵, 3×10⁵` (opacity, faintest at the smallest cap):

![Average X magnetization vs time for a 6x6 periodic Ising quench, two field strengths h (color) and five TopN term caps each (opacity)](assets/ising-quench-convergence/quench_convergence.svg)

At `h = 0.3` the three largest caps coincide up to `t = 0.8`, so the curve is converged there.
At `h = 3.1` even the two largest caps still differ by `t = 0.8`: a larger field rotation per step spreads the operator over more strings, so the same budget converges over a shorter time.
The smallest cap departs from the others by `t ≈ 0.2` at both field strengths.
Figure script: `docs/figures/ising-quench-convergence/quench_convergence.py`.

## Sections

| | |
|---|---|
| [Installation](installation.md) | install the Python package |
| [Manual](manual/index.md) | operators, circuits, propagation and measurements, read in order or by topic |
| [Examples](examples/index.md) | a first propagation, showcases, benchmarks, and comparisons with other tools |
| [Library](library/index.md) | the Python API reference |
| [Rust API](api/paulistrings/index.html) | rustdoc for the core crate, and [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) on GitHub |

Source, issues and the full research record: [github.com/lkdvos/paulistrings-rs](https://github.com/lkdvos/paulistrings-rs).
Dual-licensed MIT OR Apache-2.0.