# paulistrings-rs

[![CI](https://github.com/lkdvos/paulistrings-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/lkdvos/paulistrings-rs/actions/workflows/ci.yml)
[![docs](https://github.com/lkdvos/paulistrings-rs/actions/workflows/docs.yml/badge.svg)](https://lkdvos.github.io/paulistrings-rs/)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

<!-- Pitch paragraph is single-sourced with crates/paulistrings/README.md — keep the two word-identical. -->
Classical simulation of quantum circuits by Pauli propagation — evolving
operators in the Pauli basis under gates and noise channels, in either
the forward or Heisenberg picture. Aimed at workloads where state-vector
or tensor-network simulators are infeasible (10⁶–10⁸ terms) but the
operator stays sparse in the Pauli basis.

Inspired by [`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl).

## Highlights

- **Operator-basis Pauli propagation at 10⁶–10⁸ terms** — evolve the
  observable, not the wavefunction, in either the forward or Heisenberg
  picture.
- **GF(2)-bucketed, write-disjoint parallel engine** — layers are
  partitioned by a GF(2)-linear hash, so output buckets are statically
  predictable and never collide across threads. No global sort.
- **Open extension traits for research** — plug in a custom `Channel`
  (gate or noise model) or `TruncationPolicy` without touching the engine.
- **One core, two front ends** — the pure-Rust crate, or Python bindings
  installed via `maturin`/`pip`.
- **GPU-ready data layout** — `#[repr(C)]`, `Pod` types, fixed-fanout
  output buffers; a future GPU backend is an added kernel, not a rewrite.

## Python quickstart

```bash
./scripts/setup.sh          # one-time: creates .venv, builds the extension
source .venv/bin/activate
```

```python
import math
from paulistrings import Circuit, PauliSum

# Observable: average X magnetization on 4 qubits.
observable = PauliSum.from_strings(
    {"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4
)

circuit = Circuit(4)
circuit.rz(math.pi / 8, 0)
circuit.cnot(0, 1)
circuit.h(2)

evolved = observable.propagate(circuit)  # Heisenberg picture by default
print(evolved.expectation("x+").real)
```

## Rust quickstart

The crate is [`paulistrings`](https://docs.rs/paulistrings) on crates.io;
see its [README](crates/paulistrings/README.md) and rustdoc for the full
API. The same idea, directly against the core:

```rust
use paulistrings::{BuildAccumulator, Circuit, Direction, PauliString, Phase, propagate};
use paulistrings::{channel::Clifford1Q, truncation::TopN};
use num_complex::Complex64;

let mut acc = BuildAccumulator::<1>::new(1);
acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
let mut circuit = Circuit::<1>::new(1);
circuit.push(Clifford1Q::h(0));
let evolved = propagate(&circuit, acc.finalize(), &TopN(10), Direction::Heisenberg);
```

## Showcase

A 2D transverse-field Ising quench, evolved by Heisenberg-propagating the
average-X-magnetization observable through a Trotter circuit on 4×4 and
6×6 lattices — a regime where exact diagonalization is already infeasible
(`2^36` amplitudes for the 6×6 case) but Pauli propagation with modest
truncation finishes in seconds to minutes.

![Average X magnetization vs time for the 2D Ising quench, 4×4 and 6×6 lattices](https://raw.githubusercontent.com/lkdvos/paulistrings-rs/main/crates/paulistrings/docs/examples/img/ising_quench.svg)

Full walkthrough: [`crates/paulistrings/docs/examples/ising_2d_quench.md`](crates/paulistrings/docs/examples/ising_2d_quench.md).

## Documentation

- API reference: [docs.rs/paulistrings](https://docs.rs/paulistrings)
- Rendered rustdoc preview (rebuilt on every push to `main`):
  [lkdvos.github.io/paulistrings-rs](https://lkdvos.github.io/paulistrings-rs/)
- System design and the propagation engine: [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Repository layout

```
crates/
  paulistrings/         # core Rust library
    benches/            # criterion microbenchmarks
    examples/           # runnable end-to-end simulations
    docs/examples/      # narrative walkthroughs embedded into rustdoc
  paulistrings-py/      # PyO3 bindings (cdylib `_paulistrings`)
  membench/             # memory-bandwidth roofline probe
python/
  paulistrings/         # Python package; re-exports the extension module
benchmarks/
  python/               # pytest-benchmark suites + cross-library comparisons
  results/              # raw benchmark output (gitignored)
research/
  ideas/  plans/  notes/   # negative-result notes, hardware fact sheets, and design notes
```

## Development

```bash
./scripts/setup.sh                # one-time: creates .venv, builds the extension
source .venv/bin/activate

cargo test                        # workspace tests (Rust toolchain pinned in rust-toolchain.toml)
cargo bench -p paulistrings        # criterion microbenchmarks (release-only)

# after Rust changes, rebuild the extension before running Python tests:
maturin develop --release -m crates/paulistrings-py/Cargo.toml
pytest python/paulistrings/tests
```

## License

Dual-licensed under the [MIT License](LICENSE-MIT) or
[Apache License 2.0](LICENSE-APACHE), at your option.
