# paulistrings-rs

[![CI](https://github.com/lkdvos/paulistrings-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/lkdvos/paulistrings-rs/actions/workflows/ci.yml)
[![docs](https://github.com/lkdvos/paulistrings-rs/actions/workflows/docs.yml/badge.svg)](https://lkdvos.github.io/paulistrings-rs/)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

<!-- Pitch paragraph is single-sourced with crates/paulistrings/README.md — keep the two word-identical.
     The docs site pulls this same text in mechanically through the ANCHOR markers below
     (docs/book/src/index.md), so it cannot drift there; only these two files need syncing by hand. -->
<!-- ANCHOR: pitch -->
Classical simulation of quantum circuits by Pauli propagation — evolving
operators in the Pauli basis under gates and noise channels, in either
the forward or Heisenberg picture. Aimed at workloads where state-vector
or tensor-network simulators are infeasible (10⁶–10⁸ terms) but the
operator stays sparse in the Pauli basis.
<!-- ANCHOR_END: pitch -->

Inspired by [`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl).

## Highlights

- **Operator-basis Pauli propagation at 10⁶–10⁸ terms** — evolve the observable, not the wavefunction, in either the forward or Heisenberg picture.
- **GF(2)-bucketed, write-disjoint parallel engine** — layers are partitioned by a GF(2)-linear hash, so output buckets are statically predictable and never collide across threads.
  No global sort.
- **Partitioned across NUMA domains and MPI ranks** — split the sum by GF(2) partition rows, one pinned thread pool or one process per partition, with only the rows a layer moves across a boundary exchanged and the transfer pipelined under the layer.
- **Open extension traits for research** — plug in a custom `Channel` (gate or noise model) or `TruncationPolicy` without touching the engine.
- **One core, two front ends** — the pure-Rust crate, or Python bindings installed via `maturin`/`pip`.
- **GPU-ready data layout** — `#[repr(C)]`, `Pod` types, fixed-fanout output buffers; a future GPU backend is an added kernel, not a rewrite.

## Python quickstart

Released wheels (manylinux x86_64, macOS x86_64/arm64) are attached to [GitHub Releases](https://github.com/lkdvos/paulistrings-rs/releases) — no Rust toolchain needed:

```bash
pip install "paulistrings @ https://github.com/lkdvos/paulistrings-rs/releases/download/vX.Y.Z/paulistrings-X.Y.Z-cp39-abi3-<platform-tag>.whl"
```

where `<platform-tag>` is `manylinux_2_28_x86_64` (Linux, incl. Rusty/Popeye), `macosx_11_0_arm64` (Apple silicon) or `macosx_10_12_x86_64` (Intel Mac); one abi3 wheel per platform serves every Python >= 3.9.
Or download the `.whl` asset for your platform and `pip install ./paulistrings-*.whl`.
These wheels cover the default engine only; the `mpi` feature is never bundled into a wheel (no MPI implementation is portable across clusters) — see `CLAUDE.md` for the from-source `mpi` install.

Building from source (for contributors, or platforms without a release wheel):

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

evolved = observable.propagate(circuit, direction="heisenberg")
print(evolved.expectation("x+").real)
```

## Rust quickstart

The crate is [`paulistrings`](https://docs.rs/paulistrings) on crates.io; see its [README](crates/paulistrings/README.md) and rustdoc for the full API.
The same idea, directly against the core:

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

A larger Python examples & benchmarks suite lives under [`examples/`](examples/): a 127-qubit heavy-hex kicked-Ising cross-check against `PauliPropagation.jl` and `stim`, operator-scrambling/OTOC diagnostics, noisy utility verification, operator-backpropagation depth reduction, and stabilizer-state preparation.
Start at [`examples/README.md`](examples/README.md).

## Documentation

- Documentation site — showcases, benchmarks and cross-engine comparisons, rebuilt on every push to `main`: [lkdvos.github.io/paulistrings-rs](https://lkdvos.github.io/paulistrings-rs/).
  Source and local build instructions: [`docs/`](docs/README.md).
- API reference: [docs.rs/paulistrings](https://docs.rs/paulistrings), or the rustdoc rendered alongside the site at [lkdvos.github.io/paulistrings-rs/api/](https://lkdvos.github.io/paulistrings-rs/api/).
- System design and the propagation engine: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Repository layout

```
crates/
  paulistrings/         # core Rust library
    benches/            # criterion microbenchmarks
    examples/           # runnable end-to-end simulations
    docs/examples/      # narrative walkthroughs embedded into rustdoc
    tests/              # cross-module and differential test nets
  paulistrings-py/      # PyO3 bindings (cdylib `_paulistrings`)
  membench/             # memory-bandwidth roofline probe
python/
  paulistrings/         # Python package; re-exports the extension module
examples/
  common/               # circuit builders, oracles, timing harness, report plots
  data/                 # checked-in, provenance-tagged inputs (127q coupling map, published observables)
  tests/                # showcase and example test suites
  b1_operator_scrambling/, b2_noisy_verification/, b5_operator_backpropagation/,
  b6_resource_probes/, b7_stabilizer_prep/  # one Part-B showcase per directory — see examples/README.md
  xxz_chain/            # Benchmark D — Part A's fourth benchmark lives here, not under benchmarks/
benchmarks/
  python/               # pytest-benchmark suites + cross-library comparisons (Part A benchmarks), tests/ alongside
  julia/                # subprocess-driven PauliPropagation.jl baseline
  results/              # raw benchmark output (gitignored)
docs/
  book/                 # mdBook source for the documentation site (showcases, benchmarks,
                        #   comparisons); rendered to docs/book/site/ (gitignored)
  figures/              # source figures the book's assets are synced from
  sync-assets.sh        # links the site's figures to the committed SVGs they came from
research/
  FINDINGS.md           # one entry per experiment: question, verdict, the number that matters
  HARDWARE.md           # measured host facts (bandwidth ceilings, roofline tables, node types)
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
