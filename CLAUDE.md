# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`paulistrings-rs` implements **Pauli propagation**: classical simulation by evolving operators in the Pauli basis under
gates and noise channels — forward, or in the Heisenberg picture by applying adjoints in reverse — with truncation keeping
the sum tractable. It is not a state-vector, tensor-network, stabilizer, or matrix-product-state simulator; those are
explicit non-goals. The one storage type is a bucketed `PauliSum<W>`: per-bucket structure-of-arrays `x`/`z`/coefficient
columns partitioned by a GF(2)-linear hash `h(v) = H·v`, which makes a channel's output buckets statically predictable and
deduplication bucket-local, so no global sort exists in the propagation loop. The engine's unit of parallel work is a coset
of `span(h(D))`, write-disjoint by construction — no atomics, no locks, no synchronization inside a layer. The pure-Rust
core takes `W` as a const generic; the PyO3 bindings monomorphize widths `{1, 2, 4, 8, 16}` (64–1024 qubits), dispatching
once outside any hot loop.

`ARCHITECTURE.md` is the design source of truth; code comments cite its named sections as `ARCHITECTURE.md §Engine`. Do
not rename its `##` headings without sweeping those citations.

## Commands

Setup creates `./.venv` and builds the PyO3 extension; the Rust toolchain is pinned in `rust-toolchain.toml` (1.94.0 +
rustfmt + clippy). `PYTHON` defaults to `/usr/bin/python3.11`, absent on most Flatiron hosts — take one from Lmod instead:

```bash
module load modules/2.4-20250724 python/3.11.11   # only if /usr/bin/python3.11 is missing
PYTHON=$(which python3.11) ./scripts/setup.sh     # otherwise just ./scripts/setup.sh
source .venv/bin/activate
```

```bash
cargo build --release
cargo test                         # workspace tests
cargo test -p paulistrings         # core only
cargo test -p paulistrings <name>  # single test by substring
cargo bench -p paulistrings        # criterion microbenchmarks (release-only)
```

```bash
maturin develop --release -m crates/paulistrings-py/Cargo.toml   # rebuild after any Rust change
pytest python/paulistrings/tests
pytest benchmarks/python --benchmark-only --benchmark-json=benchmarks/results/py.json
```

`maturin develop` installs the extension into the active venv as `paulistrings._paulistrings`; `[tool.maturin]` in
`pyproject.toml` wires `python-source = "python"` and that module name, and `python/paulistrings/` re-exports it. The
release profile uses `lto = "fat"` and `codegen-units = 1` — debug builds are dramatically slower for this workload, so
always benchmark `--release`, and prefer it whenever reproducing performance behavior.

## Progress logging

The library logs through the `log` facade under the target `paulistrings::propagate`: INFO on entry and exit of each
`propagate` call, DEBUG once per layer (channel name, terms in/out, milliseconds). From Rust, install `env_logger` and set
`RUST_LOG=paulistrings=debug`. From Python the records reach stdlib `logging` via `pyo3-log` on the logger
`paulistrings.propagate`; call `paulistrings.reset_log_cache()` after changing levels mid-process, since `pyo3-log` caches
each logger's effective level.

## Testing & TDD policy

Development is test-driven: **red** (the smallest failing test that pins the behavior — unit tests in `#[cfg(test)] mod
tests` beside the code, cross-module behavior in `crates/paulistrings/tests/`), **green** (the minimum to pass, nothing
speculative), **refactor** (only after green).

- Tests assert hand-computed expected values, not the output of another unimplemented function. Where a reference exists
  (`XZ = -iY`, `X` anticommutes with `Z`), encode it as a test. Parameterize multi-qubit / multi-word logic over
  `W ∈ {1, 2}` so the const-generic surface is exercised.
- Property tests (`proptest`) for algebraic laws: multiplication associativity, the sortedness/uniqueness invariant after a
  merge, idempotence of `truncate(0.0)`.
- The differential oracle for engine work is `test_support::naive_apply_layer`, a direct `Channel::apply` loop independent
  of the bucketed path. It and the other shared helpers (seeded random-sum fixtures, comparison asserts) live in
  `crates/paulistrings/src/test_support.rs`, compiled by the `test-utils` feature via the crate's self-dev-dependency —
  add helpers there rather than copy-pasting fixtures between test files.
- No `#[ignore]`d tests. `cargo test --workspace` must be green at every commit; `cargo test --workspace --release` runs the
  same suite against the shipping codegen. Benchmarks follow tests, never the reverse.
- Commit logical units, and check in with the user at feature boundaries rather than rolling several features into one commit.

## Determinism policy

Bitwise output preservation is **not required — anywhere, for anything**. The correctness bar is agreement to floating-point
tolerance (`test_support::assert_terms_close`); equal-key summation order is unspecified and free to change between
configurations and optimizations. Tests that pin exact output bits — the fingerprint net
(`engine/bucketed.rs::layer_fingerprints_are_stable`), the thread-count and bucket-count byte-identity tests — are
convenience tripwires for *unintended* perturbation: when one trips under a change that is correct to tolerance, regenerate
its literals or demote it to `assert_terms_close` in the same commit, with a one-line note. Never design, constrain, or
reject an optimization to keep output bits stable.

## Performance discipline

- Benchmark `--release` only; keep input generation deterministic (seeded RNG) outside the timed region; report single-thread
  and multi-thread numbers separately. The probe is
  `cargo run --release --features phase-timing --example phase_breakdown`; its `phase-timing` feature (`engine/stats.rs`)
  is measurement-only and never in the default set.
- `scripts/bench-campaign.sh` plus `benchmarks/PROFILING.md` is the canonical change → measure → compare loop; output lands
  in the gitignored `benchmarks/results/<date>-<host>/` with commit, CPU, rustc version and thread count in the provenance
  header. Run campaigns with `RUST_LOG` unset — with no logger installed the per-layer logging is one static level check
  and allocates nothing, whereas an enabled `debug` filter adds a clock read per layer.
- Single-shot campaign noise on the reference host is ±5–8% single-threaded and ±10–26% at 8–32 threads — untouched code
  moves that much between campaigns. Smaller effects need `scripts/ab-compare.sh` (two prebuilt binaries alternated
  adjacent in time, paired per-run deltas); its acceptance criterion is **direction consistency across every pair**, with
  the median Δ% as the effect size. Pairs disagreeing in sign mean "no consistent change" — not a small win, not a trend.
- Roofline denominators come from `crates/membench` + `scripts/bandwidth.sh`; the reference host's measured ceiling is the
  fact sheet `research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`.
- LTO code-layout effects are real: the `#[inline]` set in `engine/merge.rs` is A/B-verified load-bearing in both directions
  (the hint on `sort_rows_with_scratch` is worth ~6%; adding one to `merge2_into` cost +20–34%). Read the comments there
  before adding or removing an attribute.

## Known gaps

- Everywhere a Pauli string is parsed or read, the convention is Hermitian: a coefficient multiplies the literal Pauli
  string, and `Y` maps to the symplectic key `(x=1, z=1)` with no phase factor. Phases arise only from products
  (`mul_assign` returns `i^k` for the caller to fold). History of the one convention conflict this repo had:
  `research/notes/2026-08-31-python-test-triage.md` (resolved).
- `PauliSum::from_strings` is `pub(crate)` + `#[cfg(test)]`, so Rust tests build sums through it or `BuildAccumulator`.
- A channel with support on more than `MAX_LOCAL_SUPPORT = 2` qubits (other than `PauliRotation`, which overrides
  `prepare` at any generator weight) makes `propagate` **panic** — there is no fallback path. Generalization design in
  `research/notes/2026-08-31-local-ptm-generalization.md`.

## Repo layout

- `crates/paulistrings/` — pure Rust core, no Python deps. Modules: `pauli_string`, `phase`, `pauli_sum`,
  `bucket/{hash,sum}`, `accumulator`, `circuit`, `channel/{clifford,rotation,unitary,noise,identity,prepared}`,
  `truncation/{builtin}`, `engine/{bucketed,coset,merge,stats}`, `test_support`, `examples`; re-exports in `lib.rs`. Also
  `benches/pauli_ops.rs` (criterion), runnable `examples/`, and walkthroughs in `docs/examples/`.
- `crates/paulistrings-py/` — PyO3 bindings, cdylib `_paulistrings`, abi3-py39, pyo3 0.22. Modules: `sum`, `circuit`,
  `gates`, `noise`, `truncation`, `channel_spec`, `truncation_spec`, `macros`.
- `crates/membench/` — STREAM-style memory-bandwidth probe behind `scripts/bandwidth.sh`. `python/paulistrings/` — the
  Python package shipped to users: a thin re-export of the extension, `interop.py` (stim/qiskit/task-JSON circuit
  importers) and `io.py` (`.npz` save/load), plus `tests/`.
- `benchmarks/` — `python/` (pytest-benchmark suites, cross-library comparisons against `qiskit.SparsePauliOp`,
  `openfermion.QubitOperator`, `stim.PauliString`, plus the suite's Part A benchmarks A/B/C/E — Benchmark D lives in
  `examples/xxz_chain/`), `julia/` (subprocess-driven baseline against `PauliPropagation.jl`, pinned version, schema-v1
  task JSON, out of CI), `PROFILING.md`, gitignored `results/`. `scripts/` — setup, campaign, A/B, profiling,
  perf-counter, bandwidth, topology and reporting tooling.
- `examples/` — the Python examples & benchmarks suite's showcases (Part B): `common/` (circuit builders, oracles,
  timing harness, report plots), `data/` (checked-in, provenance-tagged inputs), one directory per showcase. See
  `examples/README.md`.
- `research/` — `plans/`, `notes/`, named `YYYY-MM-DD-short-slug.md`; `notes/` holds negative-result write-ups,
  hardware fact sheets, and forward design notes; `plans/` holds execution plans (e.g. the examples & benchmarks suite).
  Nothing here is load-bearing for the build. **Read the negative-result notes in `research/notes/` before
  re-attempting an optimization idea:** `2026-08-26-why-s5-concatenation-fails.md`
  (support-bit bucket concatenation cannot replace a sort), `2026-08-31-v0.6-results.md` (three rejected gather/merge
  variants — recompute-in-merge borrowing, segment-copy merging, interleaved transient key layout),
  `2026-08-30-static-coset-placement.md` (static coset→worker placement, 1.25–1.9× slower than work-stealing), and the
  bandwidth-ceiling fact sheet above.
