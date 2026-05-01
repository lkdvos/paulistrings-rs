# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This repo is **v0.1 scaffolding** — the module layout, types, and trait surfaces match the design document, but most algorithm bodies are `todo!()`. The shapes (`PauliString<W>`, `PauliSum<W>`, `Channel`, `TruncationPolicy`, `Direction`, `propagate`, `BuildAccumulator`, the sort-merge engine skeleton) are in place so cross-module signatures compile; the implementation order from §13 of the design doc governs which `todo!()` to fill in next. Treat `research/plans/2026-04-30-v0.1-scope.md` as the source of truth for architecture; the stub bodies reference its section numbers (`§3.1`, `§5`, etc.) so the mapping back to the doc is direct.

Real (non-stub) code so far: `PauliString` `Ord`/`Hash`/`Pod`/`Zeroable` + `identity`; `PauliSum::{empty, len, num_qubits, x, z, coeff, assert_invariants}`; the `And`/`Or` truncation combinators; `CoefficientThreshold::keep_term`; the Python width-dispatch enums and module registration. Everything else is a typed placeholder.

One deliberate departure from the doc: `Channel::MAX_FANOUT` is a method `max_fanout(&self) -> usize`, not an associated `const`. This is required for `Box<dyn Channel<W>>` storage in `Circuit` (the channel set is open for user extensions, §6). Concrete impls return literal constants, so call sites through generics still constant-fold.

## Commands

First-time / fresh-clone setup (creates `./.venv` and builds the PyO3 extension):
```bash
./scripts/setup.sh
source .venv/bin/activate
```
The Rust toolchain is pinned in `rust-toolchain.toml` (1.94.0 + rustfmt + clippy); `rustup` honors it automatically. Python defaults to `/usr/bin/python3.11`; override with `PYTHON=...`.

Rust:
```bash
cargo build --release
cargo test                         # workspace tests
cargo test -p paulistrings         # core only
cargo test -p paulistrings <name>  # single test by substring
cargo bench -p paulistrings        # criterion microbenchmarks (release-only)
```

Python (in a venv with `maturin` installed):
```bash
maturin develop --release -m crates/paulistrings-py/Cargo.toml
pytest python/paulistrings/tests
pytest benchmarks/python --benchmark-only --benchmark-json=benchmarks/results/py.json
```

`maturin develop` builds the Rust extension and installs it into the active venv as `paulistrings._paulistrings`. The Python package at `python/paulistrings/` re-exports it; `[tool.maturin]` in `pyproject.toml` wires `python-source = "python"` and `module-name = "paulistrings._paulistrings"`.

The release profile uses `lto = "fat"` and `codegen-units = 1` — debug builds are dramatically slower for this workload, so always benchmark `--release` and prefer `--release` when reproducing performance behavior.

## Architecture (big picture)

The library implements **Pauli propagation**: classical simulation by evolving operators in the Pauli basis under gates and noise channels (Heisenberg-picture or forward). It is not a state-vector, tensor-network, stabilizer, or MPS simulator — those are explicit non-goals.

Four design pillars, in priority order: (1) correctness of the Pauli algebra, (2) performance at 10⁶–10⁸ terms, (3) extensibility for research (custom channels and truncation), (4) GPU-readiness as a future backend without restructuring.

### Core data types

- **`PauliString<const W: usize>`** — symplectic encoding `(x: [u64; W], z: [u64; W])` with `I=(0,0), X=(1,0), Z=(0,1), Y=(1,1)`. One word covers 64 qubits. `Copy + Pod + Zeroable`, `#[repr(C)]` with no padding so it's GPU-uploadable. The load-bearing trait is **`Ord`** (lex over concatenated words), not `Hash` — the engine is sort-based, not hashmap-based. There is no stored phase: `mul_assign` returns the `i^k` phase as a `u8` in `0..4`, and callers fold it into their `Complex64` coefficient at the boundary (e.g. when inserting into a `PauliSum` or `BuildAccumulator`).
- **`PauliSum<const W: usize>`** — structure-of-arrays: parallel `Vec<[u64; W]>` for `x` and `z`, `Vec<Complex64>` for coefficients, plus `num_qubits`. **Invariant: sorted by `(x, z)` key, no duplicates.** SoA is chosen so coefficient-only and key-only scans get full cache utilization, and so each `Vec` maps directly to a GPU device buffer.

### The central algorithm: sort-merge (not hashmap)

When a channel acts, it perturbs only the bits in its support. A sorted input therefore stays *almost sorted* — order is preserved outside the support, perturbed only inside. Each layer is a three-phase pipeline:

1. **Scan** — for each input term, channel writes ≤ `MAX_FANOUT` outputs into a pre-allocated buffer of size `n_in × MAX_FANOUT`. Embarrassingly parallel.
2. **Bucket** — partition outputs into `2^(2|support|)` buckets indexed by support bits (4 buckets for 1Q, 16 for 2Q). Within each bucket relative order is inherited, so concatenation is sorted.
3. **Merge** — segmented reduction: adjacent equal keys combine, truncation `keep_term` is folded in.

CPU and GPU implementations share this structure; only the parallelism mechanics differ (Rayon chunked scans vs CUB primitives). All parallelism is shared-nothing — no concurrent hashmap, no per-phase synchronization.

### Extensibility traits

- **`Channel<W>`** — `support()`, const `MAX_FANOUT`, `apply(input, coeff, &mut OutputBuffer)`. The engine reads `support` for bucket layout and `MAX_FANOUT` to size buffers without dynamic growth (also a hard requirement for GPU kernels). Built-ins planned: `Clifford1Q/2Q`, `PauliRotation`, `GeneralUnitary1Q/2Q`, `Depolarizing`, `Dephasing`, `AmplitudeDamping`.
- **`TruncationPolicy<W>`** — split into a hot `keep_term` (per-output, must inline to nanoseconds) and `finalize_layer` (once per layer, may be non-local). Built-ins: `CoefficientThreshold`, `WeightCutoff`, `TopN`. Compose via `And`/`Or`, exposed in Python as `&` / `|`.

These two traits are the research extension points.

### Width monomorphization

`W` is a const generic in Rust, but Python passes `num_qubits` at runtime. The Python boundary monomorphizes at a fixed set `{1, 2, 4, 8, 16}` (= 64, 128, 256, 512, 1024 qubits) and dispatches via a single match on `PauliSumImpl` outside any hot loop. Trades binary size for fully-unrolled bit ops. Rust users hitting the core directly choose their own `W`.

### Ingestion vs propagation

`BuildAccumulator<W>` (hashmap with `FxBuildHasher`, since Pauli bitstrings are already high-entropy) is the **ingestion path only** — Hamiltonian parsing, dict-construction, etc. `finalize()` produces a sorted/deduped `PauliSum`. It is **not** used inside the propagation loop; that's strictly sort-merge. There is, however, a planned small-sum fast path that falls back to hashmap merging below an empirical threshold.

## Testing & TDD policy

Development is test-driven. For each slice in
`research/plans/2026-04-30-v0.1-tdd-slices.md`:

1. **Red.** Write the smallest failing test that pins down the behavior.
   Unit tests live next to the code in `#[cfg(test)] mod tests`. Cross-module
   behavior goes in `crates/paulistrings/tests/`.
2. **Green.** Implement the minimum to pass — `todo!()` becomes real code,
   nothing more. Don't pre-build helpers for slices not yet started.
3. **Refactor.** Only after green. No speculative generalization.

Conventions:
- Tests assert against hand-computed expected values, not against another
  `todo!()`-implemented function. Where a reference exists (Pauli algebra
  identities like `XZ = -iY`, `X` anticommutes with `Z`), encode it as a test.
- For multi-qubit / multi-word logic, parameterize tests over `W ∈ {1, 2}`
  so the const-generic surface is exercised early.
- Property tests (via `proptest`) for algebraic laws once a slice has more
  than ~5 example tests: associativity of multiplication, sortedness
  invariant after merge, idempotence of `truncate(0.0)`, etc. Add `proptest`
  as a dev-dependency only when the first property test is written —
  not before.
- Each slice's PR/commit lands tests + impl together; do not merge a slice
  whose tests are `#[ignore]`d. Remove the `#[ignore]` placeholders in
  `crates/paulistrings/tests/pauli_string.rs` as their slices land.
- `cargo test` (workspace) must be green at every slice boundary.
- Benchmarks (`cargo bench`) follow tests, not the other way around — never
  tune a `todo!()`.
- **At every phase boundary** (the Phase 1–11 groupings in
  `research/plans/2026-04-30-v0.1-tdd-slices.md`): pause, summarize what
  landed, and check in with the user before continuing to the next phase.
  Commit the phase as a single logical unit only after the user confirms.
  Do not roll multiple phases into one commit, and do not start the next
  phase before the current one is committed.

## Repo layout

- `crates/paulistrings/` — pure Rust core (no Python deps). Has `benches/pauli_ops.rs` (criterion). Source modules: `pauli_string`, `pauli_sum`, `accumulator`, `circuit`, `channel/{clifford,rotation,unitary,noise}`, `truncation/{builtin}`, `engine/{sort_merge}`. Top-level re-exports in `lib.rs`.
- `crates/paulistrings-py/` — PyO3 bindings, cdylib named `_paulistrings`, abi3-py39, pyo3 0.22. Source modules: `sum`, `circuit`, `gates`, `noise`, `truncation`. `PauliSumImpl` / `CircuitImpl` enums dispatch over widths `{1, 2, 4, 8, 16}` (= 64–1024 qubits).
- `python/paulistrings/` — Python source tree shipped to users; thin re-export of the extension.
- `benchmarks/python/` — pytest-benchmark suites; cross-library comparisons against `PauliStrings.jl`, `qiskit.SparsePauliOp`, `openfermion.QubitOperator`, `stim.PauliString` (where applicable). `benchmarks/results/` is gitignored.
- `research/` — `ideas/`, `plans/`, `notes/`, `literature/`. Naming convention `YYYY-MM-DD-short-slug.md`. Nothing here is load-bearing for the build.

## Benchmark conventions

- Always `--release` / `cargo bench`. Never benchmark debug builds.
- Keep input generation deterministic (seeded RNG) and outside the timed region.
- Report single-thread and multi-thread numbers separately for parallel ops.
- When recording results to `benchmarks/results/<date>-<machine>/`, include commit hash, CPU, RAM, compiler version, BLAS (if any), thread count.
