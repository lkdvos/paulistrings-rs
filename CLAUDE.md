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
once outside any hot loop. Above that sits an optional **partitioned** engine (`ARCHITECTURE.md §Partitioning`): the sum
split across `P ≤ 16` NUMA domains by designated GF(2) partition rows, one pinned Rayon pool each, with a push-model
exchange of the rows a layer moves across domains and a `Transport` trait as the seam. The same layer loop runs
distributed — one partition per MPI rank (`DistributedSum`, the off-by-default `mpi` feature) — with the transport as
the only difference.

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

The probe takes the partitioned engine through the same flags as everything else — `--partitions` is a comma list, and
`--partition-cpus` one CPU list per partition, semicolon-separated (`scripts/host-topology.sh` exports a `PARTITION_CPUS`
map per known host):

```bash
cargo run --release --features phase-timing --example phase_breakdown -- \
  --partitions 2 --partition-cpus "0-7,16-23;8-15,24-31" --threads 32 --n 1000000 --layers su4
```

The MPI transport is behind the off-by-default `mpi` feature, which needs an MPI installation plus `libclang` for
rsmpi's bindgen. Every gate below must be run from a shell with the modules loaded:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7   # sets MPICC
export LIBCLANG_PATH=$(llvm-config --libdir)                 # bindgen (rsmpi)

cargo test -p paulistrings --features mpi        # includes tests/mpi_ranks.rs as a one-rank world
scripts/mpi-test.sh --ranks 2,4 [--release]      # the same net under mpirun
cargo clippy -p paulistrings --all-targets --features mpi -- -D warnings
```

Quiet-box campaigns run on an exclusive Slurm node from the templates in `scripts/slurm/` (`ab-campaign.sbatch`,
`mpi-ranks.sbatch`). **Submitting is the user's step, never an agent's** — write or adjust the template and hand over the
`sbatch` line.

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
- The partitioned engine has its own differential nets: `tests/propagate_partitioned.rs` (gathered output against
  `propagate`, over the built-in channel set × direction × partition count) and the per-layer matrices in
  `engine/partitioned/layer.rs`. Every test configuration uses `Placement::Unpinned`, so the suite runs on a one-node
  box or a `taskset`ed CI container; placement itself is covered by `topology.rs`'s own tests. A policy used in a
  partitioned test must answer `finalizes_layer() == false` or implement `PartitionedTruncation` — the trait default
  panics on a policy that finalizes layers with no collective form.
- The distributed driver has two nets: `tests/propagate_distributed.rs` runs `DistributedSum` over
  `InProcessTransport` (one thread per "rank", no MPI, so it is part of the default `cargo test`), and
  `tests/mpi_ranks.rs` runs the same matrix over `MpiTransport` under `mpirun`. The latter is `harness = false` —
  the cases are collective, so they must run in one order on every rank — and all-reduces each verdict so every
  rank exits with the same status.
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
reject an optimization to keep output bits stable. The partitioned engine adds one tripwire of the same kind:
`propagate_partitioned` at `P = 1` is byte-identical to `propagate`, and across partition counts the bar is tolerance.

## Performance discipline

- **Every cargo feature is off by default and stays that way** — `phase-timing`, `test-utils` and `mpi` alike. The
  default build must be byte- and performance-identical to one from before the feature existed, which for `mpi`
  also means `crates/paulistrings/build.rs` emits nothing at all without `CARGO_FEATURE_MPI`.
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
- **A partitioned cell (`P > 1`) runs under no placement prefix at all** — `numactl --membind` forces every page onto one
  node and `taskset`/`--cpunodebind` shrink the mask the engine's own `Auto` placement reads, both defeating the split.
  The engine's pinning is the only pinning in effect; take the CPU lists from `scripts/host-topology.sh`'s
  `PARTITION_CPUS`. `P = 1` under the existing `node0`/`phys8` placements is the one-socket reference.
- P=1 vs P=2 is a **runtime-knob** A/B, not a code A/B: `scripts/ab-compare.sh --probe-b '<args with --partitions 2>'`
  runs one binary both ways and pairs on `(layer, threads)`.
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
- Partitioned mode rejects exact `TopN` at compile time (a distributed `k`-th selection has no collective form yet);
  `ApproxTopN` is partition-exact and is the partitioned default. Thread and memory pinning are Linux-only — elsewhere
  the topology module reports one node and pins nothing, so a partitioned run is correct but unplaced.
- The `mpi` feature does not build without an MPI installation and a `libclang` for rsmpi's bindgen (`LIBCLANG_PATH`),
  which is why `[package.metadata.docs.rs]` names its features explicitly instead of `all-features = true`. A
  distributed run is one partition per rank (`D = 1`), placed by the launcher's affinity mask; there is no
  domains-per-rank hybrid, the rank count must be a power of two, the input must be replicated on every rank, and
  the wire format is raw host bytes (same architecture and same `W` on every rank).
- Partition rows are drawn at random by default, so export volume is a property of the draw: roughly half of a dense
  two-qubit gate's deltas cross at `P = 2`. Tuning the rows (cut-like rows, conserved quantities) is open research.
- The debug `paulistrings` test binary aborts with `fatal runtime error: stack overflow` in roughly 1 run in 4 under
  full parallelism — pre-existing, reproduced before any partitioned code, never with a 16 MiB stack. `.cargo/config.toml`
  sets `RUST_MIN_STACK = "16777216"` as the workaround; root cause is still open.

## Repo layout

- `crates/paulistrings/` — pure Rust core, no Python deps. Modules: `pauli_string`, `phase`, `pauli_sum`,
  `bucket/{hash,sum}`, `accumulator`, `circuit`, `channel/{clifford,rotation,unitary,noise,identity,prepared}`,
  `truncation/{builtin}`, `engine/{bucketed,coset,merge,stats}`,
  `engine/partitioned/{topology,transport,plan,export,layer,truncation,runtime,driver,distributed,trace,mpi}`,
  `stabilizer`, `test_support`, `examples`; re-exports in `lib.rs` (`mpi` is re-exported as `paulistrings::mpi`).
  `build.rs` exists only for the `mpi` feature: it re-adds the `-Wl,-rpath` rsmpi drops and stamps the MPI version
  the build probed. Also
  `benches/pauli_ops.rs` (criterion), runnable `examples/`, `tests/mpi_ranks.rs` (the multi-rank net, run by
  `scripts/mpi-test.sh`), and walkthroughs in `docs/examples/`. Dependencies worth
  knowing: `libc` is a `cfg(target_os = "linux")` target dependency (pinning and `set_mempolicy` only), and num-complex
  carries the `bytemuck` feature so exchange blocks cast coefficient columns to bytes without a copy.
- `crates/paulistrings-py/` — PyO3 bindings, cdylib `_paulistrings`, abi3-py39, pyo3 0.22. Modules: `sum`, `circuit`,
  `gates`, `noise`, `truncation`, `channel_spec`, `truncation_spec`, `macros`.
- `crates/membench/` — STREAM-style memory-bandwidth probe behind `scripts/bandwidth.sh`. `python/paulistrings/` — the
  Python package shipped to users: a thin re-export of the extension, `interop.py` (stim/qiskit/task-JSON circuit
  importers) and `io.py` (`.npz` save/load), plus `tests/`.
- `benchmarks/` — `python/` (pytest-benchmark suites, cross-library comparisons against `qiskit.SparsePauliOp`,
  `openfermion.QubitOperator`, `stim.PauliString`, plus the suite's Part A benchmarks A/B/C/E — Benchmark D lives in
  `examples/xxz_chain/`), `julia/` (subprocess-driven baseline against `PauliPropagation.jl`, pinned version, schema-v1
  task JSON, out of CI), `PROFILING.md`, gitignored `results/`. `scripts/` — setup, campaign, A/B, profiling,
  perf-counter, bandwidth, topology and reporting tooling, plus `slurm/` (user-submitted `sbatch` templates for the
  partitioned-engine campaigns on an exclusive node).
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
