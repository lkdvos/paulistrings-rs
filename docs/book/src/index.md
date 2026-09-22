# paulistrings-rs

{{#include ../../../README.md:pitch}}

Inspired by [`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl);
compared, term for term, against
[`PauliPropagation.jl`](https://github.com/MSRudolph/PauliPropagation.jl) — see
[Against other tools](examples/comparisons.md).

## Quickstart

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

An observable, a circuit, a propagation, a readout — the [Manual](manual/index.md) opens with this same run and explains each line; [First propagation](examples/first-propagation.md) carries it further, into term inspection and a validation sweep.

![Average X magnetization vs time for the 2D Ising quench, 4×4 and 6×6 lattices](assets/ising-quench/ising_quench.svg)

A 2D transverse-field Ising quench, computed by Heisenberg-propagating the
average-X-magnetization observable through a Trotter circuit — a regime where
exact diagonalization is already infeasible (`2^36` amplitudes for the 6×6
lattice) but Pauli propagation with modest truncation finishes in seconds to
minutes. Setup, truncation and error bar:
[the 2D Ising quench](examples/index.md#the-2d-ising-quench),
which links on to the crate's full Rust walkthrough.

## Pauli propagation

Write the observable, not the state, in the Pauli basis:

```text
O = Σ_P c_P P ,      P ∈ {I, X, Y, Z}^n
```

and evolve *it*. Each gate maps every Pauli string to a short sum of Pauli
strings — one string for a Clifford gate, two for a Pauli rotation
`exp(-iθP/2)`, a rescale for most noise channels — so a circuit layer is a
fan-out over the terms followed by a deduplicating merge. In the Heisenberg
picture the channel list is walked in reverse and each channel's adjoint is
applied, giving `U†OU`; in the forward picture it is walked as written, giving
`UOU†`. The expectation value against a product state is then one masked pass
over the surviving terms — never an expansion over `2^n` amplitudes.

The cost is not the qubit count. It is the number of Pauli strings the operator
spreads over, which grows with circuit depth until **truncation** holds it: a
coefficient threshold, a Pauli-weight cap, a top-`k` budget. What every result on
this site therefore has to report is how much of the operator the truncation
deleted, and whether the answer still moves when the cutoff is tightened. Every
showcase and benchmark page answers that with a convergence sweep, and says so
when the point is *not resolved*.

## Scope

- **Operator-basis Pauli propagation at 10⁶–10⁸ terms**, in either picture.
- **A GF(2)-bucketed, write-disjoint parallel engine.** Terms are partitioned by
  a GF(2)-linear hash `h(v) = H·v`, which makes a channel's output buckets
  statically predictable and deduplication bucket-local — so the unit of
  parallel work is a coset that no other worker writes to. No atomics, no locks,
  no global sort in the propagation loop.
- **Open extension points for research.** A custom gate, noise model or
  truncation policy plugs in without touching the engine; implementing one is
  a Rust-side hand-off — see the rustdoc linked below.
- **Handles 64–1024 qubits via compile-time width tiers**, picked
  automatically from the qubit count, with dispatch done once outside the hot
  loop.
- **A GPU-ready, C-compatible plain-data layout** with fixed-fanout output
  buffers: a future GPU backend is an added kernel, not a rewrite.

## Non-goals

State-vector, tensor-network, stabilizer and matrix-product-state simulation
are **explicit non-goals**. This engine has one storage type — a bucketed sum
of parallel x/z/coefficient columns — and one loop.
[Against other tools](examples/comparisons.md) says which method fits which
problem, including the two places this engine is measurably *slower* than the
alternative.

Two hard edges worth knowing before you start:

- A channel with support on more than two qubits (other than a Pauli rotation,
  which handles any generator weight) makes `propagate` **panic**. There is no
  fallback path.
- A truncated Pauli sum has **no variational bound**. Discarded terms carry
  signs, so a partial sum can sit on either side of the truth and the error
  need not be monotone in the cutoff. This is measured, not hypothetical —
  [Benchmark B](examples/benchmarks/b-theta-sweep.md#non-monotone-truncation-error)
  and [Benchmark C](examples/benchmarks/c-deep-trotter.md) both show it
  happening.

## Sections

| | |
|---|---|
| [Installation](installation.md) | install the Python package |
| [Manual](manual/index.md) | operators, circuits, engine and propagation, measurements — read start to finish, or by topic |
| [Examples](examples/index.md) | a guided first propagation, measured showcases and benchmarks, and comparisons against other tools |
| [Library](library/index.md) | this book's Python API reference |
| [Rust API](api/paulistrings/index.html) | rustdoc for the core crate, plus [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) on GitHub — the hand-off point for Rust users, not part of this book |

## Numbers on this site

Every number here is copied from a **committed** results file or README in the
repository, and every page names the file it came from. No measurement was
taken to build this site. Three consequences:

- **Wall times are indicative, not campaign-grade.** They were taken on a
  shared workstation (Intel Xeon Gold 6244 @ 3.60 GHz, `ccqlin038`) whose stated
  single-thread run-to-run noise is ±5–8%, and ±10–26% at 8–32 threads. Term
  counts, expectation values, parity outcomes and convergence verdicts are
  load-independent; those are the numbers to quote. Anything under ~10% needs
  the repo's A/B protocol (`scripts/ab-compare.sh`), not these tables.
- **"Not claimable" is a result.** Several pages report a configuration whose
  convergence sweep never plateaued, and therefore quote no value. That verdict
  comes from a criterion fixed in code before the run, and it is not bent to fit
  an answer.
- **Reproduction is one command per page.** Each showcase and benchmark page
  ends with the exact invocation that regenerates its figures and JSON.

## Repository

Source, issues and the full research record:
[github.com/lkdvos/paulistrings-rs](https://github.com/lkdvos/paulistrings-rs).
The design source of truth is
[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md);
the site's [Manual](manual/index.md) is its public summary.
Dual-licensed MIT OR Apache-2.0.
