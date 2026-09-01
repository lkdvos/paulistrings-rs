# Getting started

Two front ends over one core: a pure-Rust crate, and PyO3 bindings that
monomorphize the word width `W ∈ {1, 2, 4, 8, 16}` — 64 to 1024 qubits — and
dispatch once, outside any hot loop.

## Install

### Python

The repository ships a setup script that creates `./.venv` and builds the
extension into it:

```bash
git clone https://github.com/lkdvos/paulistrings-rs
cd paulistrings-rs
./scripts/setup.sh
source .venv/bin/activate
```

`scripts/setup.sh` expects a Python 3.11 at `/usr/bin/python3.11`; point it
elsewhere with `PYTHON=$(which python3.11) ./scripts/setup.sh`. The Rust
toolchain is pinned in `rust-toolchain.toml` (1.94.0), so no toolchain choice is
needed. After any change to the Rust sources, rebuild:

```bash
maturin develop --release -m crates/paulistrings-py/Cargo.toml
```

Build `--release`. The release profile uses `lto = "fat"` and
`codegen-units = 1`; a debug build of this workload is dramatically slower, and
never worth benchmarking.

The optional extras matter only for the example suite and the cross-library
benchmarks — the library itself needs nothing but `numpy`:

```bash
pip install -e ".[examples]"   # matplotlib, stim, qiskit, qiskit-aer — the oracles and plots
pip install -e ".[bench]"      # pytest-benchmark, qiskit, openfermion, stim
```

### Rust

The core crate is [`paulistrings`](https://docs.rs/paulistrings) on crates.io:

```toml
[dependencies]
paulistrings = "0.1"
num-complex = "0.4"
```

It has no Python dependency and no C dependency.

## Quickstart — Python

```python
import math
from paulistrings import Circuit, PauliSum, truncation

# Observable: average X magnetization on 4 qubits. Coefficients multiply the
# literal Hermitian Pauli string — `Y` carries no phase of its own.
observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)

circuit = Circuit(4)
circuit.rz(math.pi / 8, 0)
circuit.cnot(0, 1)
circuit.h(2)

evolved = observable.propagate(
    circuit,
    truncation.coeff(1e-10),      # drop |c| <= 1e-10 after every channel
    direction="heisenberg",       # U† O U — always pass this explicitly
)
print(len(evolved), evolved.expectation("x+").real)
```

`Circuit` methods push one gate per channel: `h s x y z cnot cz swap rz rx ry
pauli_rotation unitary_1q unitary_2q`, plus the noise channels `depolarize
dephase amplitude_damping pauli_channel depolarize2`. The `gates` and `noise`
submodules expose the same set as free factory functions when you want to build
a gate list first.

Reading a result out:

| call | what it gives |
|---|---|
| `expectation("x+" / "y+" / "z+")` | uniform product state `\|+…+⟩`, `\|+i…+i⟩`, `\|0…0⟩` — one masked pass over the terms |
| `expectation("0+1r…")` | per-qubit product state, one character per qubit, in qiskit's `Statevector.from_label` alphabet (`0 1 + - r l`) |
| `overlap(other)` | Hilbert–Schmidt overlap `tr(A†B)/2ⁿ` — for an observable against a state that is itself a Pauli sum |
| `identity_coefficient()` | the coefficient of `I…I` |
| `x_array()`, `z_array()`, `coefficients_array()` | zero-copy numpy views of the symplectic bit columns and coefficients, for analysis passes the library does not provide |
| `propagate_with_stats(...)` | `(evolved, PropagationStats)` — per-layer term counts in and out, and `peak_terms` |

`PauliSum.from_arrays(x, z, coefficients, num_qubits)` is the inverse of the
array accessors, so an analysis pass can hand a sum back to the engine; the
`paulistrings.io` module saves and loads a sum as `.npz`, and
`paulistrings.interop` imports circuits from stim, from qiskit, or from the
schema-v1 task JSON the [cross-engine comparisons](comparisons.md) use.

## Quickstart — Rust

```rust
use num_complex::Complex64;
use paulistrings::{BuildAccumulator, Circuit, Direction, PauliString, Phase, propagate};
use paulistrings::{channel::Clifford1Q, truncation::TopN};

// `W = 1` covers up to 64 qubits; the width is a const generic, so pick the
// smallest one that fits and the layout follows.
let mut acc = BuildAccumulator::<1>::new(1);
acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));

let mut circuit = Circuit::<1>::new(1);
circuit.push(Clifford1Q::h(0));

let evolved = propagate(&circuit, acc.finalize(), &TopN(10), Direction::Heisenberg);
```

`BuildAccumulator` is the ingestion path: add terms in any order, then
`finalize()` for a sorted, deduplicated `PauliSum`. Truncation policies live in
`paulistrings::truncation` (`CoefficientThreshold`, `WeightCutoff`, `TopN`, and
`And` / `Or` to compose them); channels live in `paulistrings::channel`
(`Clifford1Q`, `Clifford2Q`, `PauliRotation`, `GeneralUnitary1Q`,
`GeneralUnitary2Q`, and the noise channels). Both are open traits — a custom
`Channel` or `TruncationPolicy` plugs in without touching the engine.

The narrative walkthrough of a full simulation, from the Hamiltonian to the
figure, is
[`crates/paulistrings/docs/examples/ising_2d_quench.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/docs/examples/ising_2d_quench.md).

## Truncation basics

Truncation is what makes the method work, and it is applied **after every
channel** — not after every `propagate` call, and not after every "layer".
Three consequences follow immediately, and all three are load-bearing on the
pages that follow:

1. **Splitting a circuit is free.** `sum.propagate(a, p, d).propagate(b, p, d)`
   applies the same sequence of (apply, truncate) steps as
   `sum.propagate(a_then_b, p, d)`, so a time series costs one pass and a
   hybrid split costs nothing in accuracy — this is exactly the identity
   [Showcase B5](showcases/b5-operator-backpropagation.md) rests on.
2. **A gate is a truncation point.** Fusing two gates into one channel changes
   the answer, which is why every circuit in this repository's example suite is
   built one gate per `Circuit.push`, and why the cross-engine comparison can be
   compared *per layer* at all.
3. **A noise channel is a truncation point too.** In the Pauli basis a
   depolarizing channel is a coefficient rescale, so adding noise makes a fixed
   threshold bite harder at every depth. That is the whole mechanism of
   [Showcase B2](showcases/b2-noisy-verification.md).

The three built-in policies:

| Python | Rust | keeps |
|---|---|---|
| `truncation.coeff(eps)` | `CoefficientThreshold(eps)` | `\|c\| > eps` — note **strictly** greater; `\|c\| == eps` is dropped |
| `truncation.weight(k)` | `WeightCutoff(k)` | Pauli weight `<= k` |
| `truncation.topn(k)` | `TopN(k)` | at most `k` terms, largest `\|c\|` first |
| `a & b`, `a \| b` | `And(a, b)`, `Or(a, b)` | both / either |

Choosing one:

- **`coeff` is the default choice**, and the only knob whose sweep gives a
  convergence statement that is comparable across engines. Every convergence
  panel on this site is a `coeff` sweep.
- **`weight` is a blunt instrument.** It is genuinely useful when the causal
  cone is the constraint ([Showcase B5](showcases/b5-operator-backpropagation.md)
  uses `weight <= 6`), and measurably useless when it is not: on a degree-4 2D
  lattice at maximal entangling strength, a cap of 4 deletes 61% of the operator
  at the *second* Trotter step and buys no extra time at all
  ([B1 §4.1](showcases/b1-operator-scrambling.md#why-2d-is-the-hard-case)).
  Watch out at Clifford angles, where the operator passes through weight 30–40
  mid-circuit even when it lands on a single low-weight string, so
  `max_weight <= 8` can truncate the whole sum to zero terms
  ([Benchmark B §3.2](benchmarks/b-theta-sweep.md#clifford-endpoints)).
- **`topn` bounds memory, and changes what "converged" means.** With a fixed
  budget the error is set by the discarded tail rather than by a threshold, so a
  `topn` run cannot carry a `coeff`-sweep convergence panel. It is also banned
  from cross-engine comparisons, since `PauliPropagation.jl` has no equivalent.
- **`TopN` never splits a tie group** of exactly-equal coefficient magnitudes:
  the whole group is kept if it fits within `k`, dropped otherwise. On a
  symmetric lattice a tie group is a symmetry orbit, and truncation should
  commute with the lattice symmetry. The alternative tie rules move the 2D Ising
  quench trajectory by up to 1.7% (4×4) and 0.37% (6×6) — treat that spread as
  the honest error bar.
- **Keep `min_abs_coeff` above ~1e-12 on deep circuits.** `cos(π/2)` is
  `6.123233995736766e-17`, not zero, so at a Clifford angle every rotation
  leaves a numerically dead residual branch, and an untruncated 127-qubit
  propagation fans out without bound.

A last, important one: **a truncated Pauli sum has no variational bound.**
Dropped terms carry signs, so a partial sum can sit on either side of the truth
and the error need not fall monotonically as you tighten the cutoff. Both
[Benchmark B](benchmarks/b-theta-sweep.md#truncation-error-is-not-monotone-in-the-cutoff)
and [Benchmark D](benchmarks/d-xxz-chain.md#convergence-panels) measured
non-monotone rows. Read a convergence sweep as a *trend across the grid*, never
as a point-to-point improvement.

## Direction semantics

`direction` selects which conjugation the engine performs:

| | picture | what the engine does |
|---|---|---|
| `"heisenberg"` / `Direction::Heisenberg` | `U† O U` | walks the channel list **in reverse** and applies each channel's adjoint |
| `"forward"` / `Direction::Forward` | `U O U†` | walks the channel list **as written** and applies each channel |

**Always pass it explicitly.** The Python binding accepts `direction=None` and
treats it as `"forward"`, but the two pictures answer different questions and a
default silently picks one. The repository's own cross-engine task-JSON schema
refuses to default it at all — `PauliPropagation.jl`'s default is Heisenberg
and this engine's is forward, so defaulting either way would silently choose a
picture — and the Rust `propagate` takes the `Direction` as a required
argument. Treat the Python default as a convenience for one-off exploration,
not as an idiom.

Which one you want:

- **Heisenberg** for "what does this circuit measure?" — you have an observable
  and a circuit, and want `⟨ψ|U†OU|ψ⟩` for a product-state `|ψ⟩`. This is what
  every showcase and benchmark on this site uses, because a local observable
  starts as a handful of terms and only spreads as far as its causal cone.
- **Forward** for "what does this circuit do to this operator?" — evolving a
  density matrix or a Hamiltonian in the Schrödinger picture. It starts from a
  wide operator, so the cost profile is different from the outset.

Push order interacts with direction. Under `Direction::Heisenberg` the engine
iterates channels in reverse, so pushing `ZZ` rotations before `X` rotations
gives a step operator `U = U_X · U_ZZ` and Heisenberg evolution computes
`U_ZZ† U_X† O U_X U_ZZ`. Get the order right when you build the circuit, and
the direction flag does the rest.

One asymmetry to know about: `direction="forward"` has **no counterpart in
`PauliPropagation.jl` 0.8.2** for `unitary_1q`, `unitary_2q`,
`amplitude_damping`, `pauli_channel` or `depolarize2` — that library defines no
Schrödinger transfer map for them. So every cross-engine comparison on this
site is Heisenberg. Details in [Comparisons](comparisons.md).

## Progress logging

Long runs report through the `log` facade under the target
`paulistrings::propagate`: INFO on entry and exit of each `propagate` call,
DEBUG once per layer with the channel name, terms in and out, and milliseconds.

```python
import logging
import paulistrings

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("paulistrings.propagate").setLevel(logging.DEBUG)
paulistrings.reset_log_cache()   # pyo3-log caches each logger's effective level
```

From Rust, install `env_logger` and set `RUST_LOG=paulistrings=debug`. Leave
`RUST_LOG` unset when timing: with no logger installed the per-layer logging is
one static level check and allocates nothing, whereas an enabled `debug` filter
adds a clock read per layer.

## Threads

The engine parallelizes over cosets with Rayon. **Set `RAYON_NUM_THREADS`
before the interpreter starts** — Rayon builds its global pool at the first
`propagate` call and never resizes it, so setting the variable from inside a
script that has already imported `paulistrings` does not reliably reach it:

```bash
RAYON_NUM_THREADS=1 python my_script.py
```

Every timed number on this site was taken this way; the sweeps that are physics
measurements rather than timings say so and run on the default pool.
