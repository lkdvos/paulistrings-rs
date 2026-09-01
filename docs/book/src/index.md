# paulistrings-rs

{{#include ../../../README.md:pitch}}

Inspired by [`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl);
compared, term for term, against
[`PauliPropagation.jl`](https://github.com/MSRudolph/PauliPropagation.jl) — see
[Comparisons](comparisons.md).

*(That paragraph is pulled verbatim out of the repository
[`README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/README.md) at
build time, so it cannot drift from the pitch the crate ships with.)*

![Average X magnetization vs time for the 2D Ising quench, 4×4 and 6×6 lattices](assets/ising-quench/ising_quench.svg)

A 2D transverse-field Ising quench, computed by Heisenberg-propagating the
average-X-magnetization observable through a Trotter circuit — a regime where
exact diagonalization is already infeasible (`2^36` amplitudes for the 6×6
lattice) but Pauli propagation with modest truncation finishes in seconds to
minutes. Full walkthrough:
[`crates/paulistrings/docs/examples/ising_2d_quench.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/docs/examples/ising_2d_quench.md).

## What Pauli propagation is

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
spreads over, which grows with circuit depth until **truncation** — a
coefficient threshold, a Pauli-weight cap, a top-`k` budget — holds it. So the
central question for every result on this site is not "did it run?" but "how
much of the operator did the truncation delete, and does the answer still move
when you tighten it?" Every showcase and benchmark page here answers that
explicitly, with a convergence sweep, and says so when the answer is *no,
this point is not resolved*.

## What this library is

- **Operator-basis Pauli propagation at 10⁶–10⁸ terms**, in either picture.
- **A GF(2)-bucketed, write-disjoint parallel engine.** Terms are partitioned by
  a GF(2)-linear hash `h(v) = H·v`, which makes a channel's output buckets
  statically predictable and deduplication bucket-local — so the unit of
  parallel work is a coset that no other worker writes to. No atomics, no locks,
  no global sort in the propagation loop.
- **Open extension traits for research.** A custom `Channel` (gate or noise
  model) or `TruncationPolicy` plugs in without touching the engine.
- **One core, two front ends.** The pure-Rust crate takes the word width `W` as
  a const generic; the PyO3 bindings monomorphize `{1, 2, 4, 8, 16}` (64–1024
  qubits) and dispatch once, outside any hot loop.
- **A GPU-ready data layout.** `#[repr(C)]`, `Pod` types, fixed-fanout output
  buffers: a future GPU backend is an added kernel, not a rewrite.

## What this library is not

State-vector, tensor-network, stabilizer and matrix-product-state simulation
are **explicit non-goals**. This engine has one storage type — a bucketed
`PauliSum<W>` of structure-of-arrays `x`/`z`/coefficient columns — and one
loop. Where those other methods are the right tool, they are the right tool;
[Comparisons](comparisons.md) says which is which, including the two places
this engine is measurably *slower* than the alternative.

Two hard edges worth knowing before you start:

- A channel with support on more than two qubits (other than `PauliRotation`,
  which handles any generator weight) makes `propagate` **panic**. There is no
  fallback path.
- A truncated Pauli sum has **no variational bound**. Discarded terms carry
  signs, so a partial sum can sit on either side of the truth and the error
  need not be monotone in the cutoff. This is measured, not hypothetical —
  [Benchmark B §3.3a](benchmarks/b-theta-sweep.md#truncation-error-is-not-monotone-in-the-cutoff)
  and [Benchmark C](benchmarks/c-deep-trotter.md) both show it happening.

## Start here

| | |
|---|---|
| [Getting started](getting-started.md) | install, both quickstarts, truncation, direction semantics |
| [Showcases](showcases/index.md) | four measured applications: scrambling and OTOCs, noisy verification at 127 qubits, hybrid depth reduction, operator-complexity probes |
| [Benchmarks](benchmarks/index.md) | five benchmarks A–E: setup, oracle, result — including the two honest negative results |
| [Comparisons](comparisons.md) | vs `PauliPropagation.jl` (term-for-term parity, and the measured crossover), vs state-vector and stabilizer simulators |
| [API reference](api/paulistrings/index.html) | rustdoc for the core crate, rebuilt on every push to `main` |

## How to read the numbers on this site

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
[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md).
Dual-licensed MIT OR Apache-2.0.
