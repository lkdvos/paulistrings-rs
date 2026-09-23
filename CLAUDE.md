# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

`paulistrings-rs` implements **Pauli propagation**: classical simulation by evolving operators in the Pauli basis under gates and noise channels, forward or in the Heisenberg picture, with truncation keeping the sum tractable.
State-vector, tensor-network, stabilizer and matrix-product-state simulation are explicit non-goals.

The one storage type is a bucketed `PauliSum<W>`: structure-of-arrays `x`/`z`/coefficient columns partitioned by a GF(2)-linear hash `h(v) = H·v`.
That makes a channel's output buckets statically predictable and deduplication bucket-local, so the propagation loop contains no global sort.
The unit of parallel work is a coset of `span(h(D))`, write-disjoint by construction — no atomics, no locks, no synchronization inside a layer.
The core takes `W` as a const generic; the PyO3 bindings monomorphize widths `{1, 2, 4, 8, 16}` (64–1024 qubits) and dispatch once outside any hot loop.
Above that sits an optional **partitioned** engine: the sum split across `P ≤ 64` NUMA domains (or, distributed, ranks) by designated GF(2) partition rows, one pinned Rayon pool each, with a push-model exchange and a `Transport` trait as the seam.
The same layer loop runs distributed, one partition per MPI rank (`DistributedSum`, the off-by-default `mpi` feature), with the transport as the only difference.

`ARCHITECTURE.md` is the design source of truth and code cites its named sections as `ARCHITECTURE.md §Engine`.
Do not rename its `##` headings without sweeping those citations.
`research/FINDINGS.md` records what was tried and rejected; `research/HARDWARE.md` holds the measured host facts.

## Comment & prose style

This repository was built with heavy LLM assistance and was drowning in commentary.
These rules are binding on every file, including markdown.

1. **One sentence per line.** Never reflow a paragraph across lines. A long sentence stays one long line.
2. **Comment blocks are rare.** A `//!` module header is at most five lines and says what the module contains, not how it works. A run longer than ~8 comment lines needs a reason.
3. **No history in comments.** No dates, no "we tried X", no "previously", no A/B tables, no "re-measured on <host>". That is what `research/FINDINGS.md` and git are for.
4. **Cite, don't paraphrase.** `ARCHITECTURE.md §Engine` is one line and outranks the paragraph restating it.
5. **Comment the surprise** — a safety invariant, a non-obvious algorithm choice, a contract an implementor must honour. Never the mechanics the code already states.
6. **Docs on private items are terse.** Multi-paragraph `///` belongs only on the `pub` surface that rustdoc and Python users see.

## Commands

End users install a released wheel from GitHub Releases (README's Python quickstart) — no Rust toolchain needed.
Everything below is the from-source / contributor path.

Setup creates `./.venv` and builds the PyO3 extension; the toolchain is pinned in `rust-toolchain.toml`.
`PYTHON` defaults to `/usr/bin/python3.11`, which is absent on most Flatiron hosts — take one from Lmod instead.

```bash
module load modules/2.4-20250724 python/3.11.11   # only if /usr/bin/python3.11 is missing
PYTHON=$(which python3.11) ./scripts/setup.sh
source .venv/bin/activate
```

```bash
cargo test --workspace                                           # must be green at every commit
cargo clippy --workspace --all-targets -- -D warnings
maturin develop --release -m crates/paulistrings-py/Cargo.toml   # rebuild after any Rust change
pytest python/paulistrings/tests
```

The release profile uses `lto = "fat"` and `codegen-units = 1`; debug builds are dramatically slower for this workload, so benchmark `--release` only.

The measurement probe, which also drives the partitioned engine (`--partitions` a comma list, `--partition-cpus` one semicolon-separated CPU list per partition, from `scripts/host-topology.sh`):

```bash
cargo run --release --features phase-timing --example phase_breakdown -- \
  --partitions 2 --partition-cpus "0-7,16-23;8-15,24-31" --threads 32 --n 1000000 --layers su4
```

The `mpi` feature needs an MPI installation plus `libclang` for rsmpi's bindgen, so every MPI command runs from a shell with the modules loaded:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH=$(llvm-config --libdir)

cargo test -p paulistrings --features mpi        # tests/mpi_ranks.rs as a one-rank world
scripts/mpi-test.sh --ranks 2,4 [--release]      # the same net under mpirun
scripts/mpi-test.sh --ranks 2,4 --python         # and the bindings' net
```

`mpi` is never bundled into a released wheel — no MPI implementation is portable across cluster/vendor combinations — so it stays a pip-driven source build against the loaded modules:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH=$(llvm-config --libdir)
pip install ".[dev]" --config-settings=build-args="--features mpi"
```

`--python` needs a second venv, because `./.venv` has no mpi4py and mpi4py must come from the same interpreter and MPI the modules provide.
Build it once and `--python` reuses it (`$VIRTUAL_ENV` overrides the path):

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 python-mpi/3.12.9
python3 -m venv --system-site-packages .venv-mpi   # gitignored
.venv-mpi/bin/pip install maturin pytest numpy
```

Both crates carry a `build.rs` that exists only for the `mpi` feature: `cargo:rustc-link-arg` is not inherited from a dependency, so without the py crate's copy the cdylib cannot find `libmpi.so.40` at import time.

The `cuda` feature needs no build-script support and no toolkit to compile: `cudarc` loads `libcuda` and `libnvrtc` at runtime and NVRTC compiles the kernels on first use, so only running needs the module (or `pip install nvidia-cuda-nvrtc-cu12` with its `lib` on `LD_LIBRARY_PATH`):

```bash
module load cuda/12.8.0                                          # libnvrtc at runtime
cargo test -p paulistrings --features cuda                       # unit nets + tests/propagate_gpu.rs; pass without a device
cargo clippy -p paulistrings-py --features cuda -- -D warnings
maturin develop --release --features cuda -m crates/paulistrings-py/Cargo.toml
pytest python/paulistrings/tests/test_cuda.py                    # skipped unless cuda_available()
```

Quiet-box campaigns run on an exclusive Slurm node from `scripts/slurm/`.
**Submitting is the user's step, never an agent's** — adjust the template and hand over the `sbatch` line.

## Releasing

Rust and Python release together, one version for both: `scripts/bump-version.sh X.Y.Z` bumps `Cargo.toml`'s `workspace.package.version` and `pyproject.toml`'s `[project] version` in one step; commit both together, never separately.
CI's `version-sync` job fails a PR if the two ever disagree.
Before tagging: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, the `mpi` CI job, and `python` CI job must all be green on `main`; also check `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p paulistrings --features phase-timing,test-utils`, which mirrors what docs.rs builds and is not covered by `cargo test`.
Dry-run `.github/workflows/release.yml` via `workflow_dispatch` before the real tag, to catch a wheel-matrix failure before it's user-visible.
**Pushing the tag is the user's step, never an agent's**: `git tag vX.Y.Z && git push origin vX.Y.Z` triggers `release.yml`, which checks the tag against both version files and publishes wheels (manylinux x86_64, macOS x86_64/arm64) to the GitHub Release.
Publishing the Rust crate to crates.io is a separate, manual `workflow_dispatch` of `.github/workflows/crates-publish.yml` (defaults to `--dry-run`) — run it after the wheel release for the same version, not instead of it.
Neither workflow writes a changelog; that stays unautomated today.

## Progress logging

The library logs through the `log` facade under the target `paulistrings::propagate`: INFO on entry and exit of each `propagate` call, DEBUG once per layer.
From Rust install `env_logger` and set `RUST_LOG=paulistrings=debug`.
From Python the records reach stdlib `logging` via `pyo3-log` on the logger `paulistrings.propagate`; call `paulistrings.reset_log_cache()` after changing levels mid-process, since `pyo3-log` caches each logger's effective level.

## Testing & TDD policy

Development is test-driven: red (the smallest failing test that pins the behavior), green (the minimum to pass), refactor (only after green).
Unit tests live in `#[cfg(test)] mod tests` beside the code, cross-module behavior in `crates/paulistrings/tests/`.

- Tests assert hand-computed expected values, not the output of another unimplemented function.
- Where a reference exists (`XZ = -iY`, `X` anticommutes with `Z`), encode it as a test.
- Parameterize multi-qubit and multi-word logic over `W ∈ {1, 2}` to exercise the const-generic surface.
- Property tests (`proptest`) cover the algebraic laws: multiplication associativity, the sortedness/uniqueness invariant after a merge, idempotence of `truncate(0.0)`.
- The differential oracle for engine work is `test_support::naive_apply_layer`, a direct `Channel::apply` loop independent of the bucketed path.
- Shared fixtures live in `crates/paulistrings/src/test_support.rs` behind the `test-utils` feature — add helpers there rather than copy-pasting between test files.
- The partitioned engine's differential nets are `tests/propagate_partitioned.rs` and the per-layer matrices in `engine/partitioned/layer.rs`; every test configuration uses `Placement::Unpinned` so the suite runs on a one-node box.
- The CUDA backend's differential net is `tests/propagate_gpu.rs` (`required-features = ["cuda", "test-utils"]`) against `propagate`, every device test opening with `test_support::require_cuda!()` so it returns early without a device; the bindings' net is `python/paulistrings/tests/test_cuda.py`, whose device tests skip unless `cuda_available()`.
- The distributed driver has two nets: `tests/propagate_distributed.rs` over `InProcessTransport` (part of the default `cargo test`) and `tests/mpi_ranks.rs` over `MpiTransport` under `mpirun` (`harness = false`, since the cases are collective and must run in one order on every rank).
- No `#[ignore]`d tests, and benchmarks follow tests rather than the reverse.
- Commit logical units and check in with the user at feature boundaries.

## Determinism policy

Bitwise output preservation is **not required — anywhere, for anything**.
The correctness bar is agreement to floating-point tolerance (`test_support::assert_terms_close`); equal-key summation order is unspecified and free to change.
Tests that pin exact output bits are convenience tripwires for *unintended* perturbation: when one trips under a change that is correct to tolerance, regenerate its literals or demote it to `assert_terms_close` in the same commit, with a one-line note.
Never design, constrain, or reject an optimization to keep output bits stable.
`propagate_partitioned` at `P = 1` is byte-identical to `propagate`; across partition counts the bar is tolerance.
A device run agrees with the host to tolerance and is bitwise reproducible run-to-run on one device, since no reduction uses a float atomic.

## Performance discipline

- **Every cargo feature is off by default and stays that way** — `phase-timing`, `test-utils` and `mpi` alike. The default build must be byte- and performance-identical to one from before the feature existed, which for `mpi` means `build.rs` emits nothing without `CARGO_FEATURE_MPI`.
- Benchmark `--release` only, keep input generation deterministic and outside the timed region, and report single-thread and multi-thread numbers separately.
- `scripts/bench-campaign.sh` plus `benchmarks/PROFILING.md` is the canonical change → measure → compare loop; run campaigns with `RUST_LOG` unset.
- Single-shot campaign noise on the reference host is ±5–8% single-threaded and ±10–26% at 8–32 threads, so smaller effects need `scripts/ab-compare.sh`. Its acceptance criterion is **direction consistency across every pair**, with the median Δ% as the effect size; pairs disagreeing in sign mean "no consistent change", not a small win.
- A **direction-consistent phase delta is not an effect if the total is flat** — let instruction count settle it.
- **Any constant tuned by wall-clock A/B before 2026-09-10 is suspect**: four recorded conclusions dissolved on re-measurement, all the same branch-alignment artifact (`research/FINDINGS.md`).
- A partitioned cell (`P > 1`) runs under **no placement prefix at all** — `numactl --membind` forces every page onto one node, and `taskset`/`--cpunodebind` shrink the mask the engine's `Auto` placement reads. The engine's pinning is the only pinning in effect. `P = 1` under the existing `node0`/`phys8` placements is the one-socket reference.
- P=1 vs P=2 is a runtime-knob A/B, not a code A/B: `scripts/ab-compare.sh --probe-b '<args with --partitions 2>'` runs one binary both ways and pairs on `(layer, threads)`.
- The probe's JSON sidecar carries the partition fields on every row, and the sub-phases (`append_ns`, `chunk_wait_ns`) are *contained in* the phase above rather than additional to it — never sum them into a total. Contract (a) in `benchmarks/PROFILING.md` lists the fields and is what to update when `phase_breakdown.rs::json_line` or `PhaseStats` changes.
- The JCC erratum (SKX102) costs this engine 9–13% wall on Cascade Lake, but the padding flag is a ~1% tax on every part without the erratum, so it is **not** in `.cargo/config.toml`. Every measurement script sources `scripts/jcc-rustflags.sh`, which detects the erratum from `/proc/cpuinfo` and appends the flag — **anything else that benchmarks must do the same**, and note that an exported `RUSTFLAGS` replaces the config's list wholesale.
- Roofline denominators come from `crates/membench` + `scripts/bandwidth.sh` against the ceilings in `research/HARDWARE.md`.

**Read `research/FINDINGS.md` before re-attempting an optimization idea.**
It records what was measured and rejected, including several ideas that look obviously good.

## Known gaps

- The convention is Hermitian everywhere a Pauli string is parsed or read: a coefficient multiplies the literal Pauli string, and `Y` maps to the symplectic key `(x=1, z=1)` with no phase factor. Phases arise only from products, where `mul_assign` returns `i^k` for the caller to fold.
- `PauliSum::from_strings` is `pub(crate)` + `#[cfg(test)]`, so Rust tests build sums through it or `BuildAccumulator`.
- A channel with support on more than `MAX_LOCAL_SUPPORT = 2` qubits makes `propagate` **panic**; there is no fallback path. `PauliRotation` is exempt, overriding `prepare` at any generator weight.
- Partitioned mode rejects exact `TopN` at compile time, since a distributed `k`-th selection has no collective form yet; `ApproxTopN` is partition-exact and is the partitioned default.
- The CUDA backend runs a policy only through its `TruncationPolicy::device_policy` tree: every builtin and every Python policy lowers, while a custom `TruncationPolicy` and exact `TopN` return `GpuError::Unsupported` before the first layer.
- One CUDA device per process for now: the bindings raise `NotImplementedError` on `device=` with several ordinals or an `"auto"` that sees more than one, and `device=` excludes `partitions=` and `comm=`.
- Thread and memory pinning are Linux-only; elsewhere the topology module reports one node and pins nothing, so a partitioned run is correct but unplaced.
- A distributed run is one partition per rank (`D = 1`), placed by the launcher's affinity mask. There is no domains-per-rank hybrid, the rank count must be a power of two, the input must be replicated on every rank, and the wire format is raw host bytes (same architecture and same `W` everywhere).
- Partition rows are drawn at random by default, so export volume is a property of the draw — roughly half of a dense two-qubit gate's deltas cross at `P = 2`. Tuning the rows is open research.
- The probe replicates its input on every rank, so its `vmhwm_kb` grows with rank count at constant terms per rank. That is a probe artefact; engine-side peak per rank is flat.
- The debug `paulistrings` test binary aborts with `fatal runtime error: stack overflow` in roughly 1 run in 4 under full parallelism. It is pre-existing and never reproduces with a 16 MiB stack, so `.cargo/config.toml` sets `RUST_MIN_STACK = "16777216"`; root cause is open.

## Repo layout

```
crates/paulistrings/      pure Rust core, no Python deps
  src/                    pauli_string, phase, pauli_sum, bucket/{hash,sum}, accumulator, circuit,
                          channel/{clifford,rotation,unitary,noise,identity,prepared},
                          truncation/builtin, engine/{bucketed,coset,merge,direct,stats},
                          engine/partitioned/*, engine/gpu/* (CUDA, behind `cuda`),
                          stabilizer, test_support
  tests/ benches/ examples/ docs/examples/
crates/paulistrings-py/   PyO3 bindings, cdylib `_paulistrings`, abi3-py39, pyo3 0.22
crates/membench/          STREAM-style bandwidth probe behind scripts/bandwidth.sh
python/paulistrings/      the shipped package: extension re-export, interop.py, io.py, tests/
benchmarks/               python/ suites, julia/ baseline, PROFILING.md, gitignored results/
examples/                 Python showcase suite; common/ is the shared library
docs/book/                mdBook site, published by CI; canonical for user-facing prose
research/                 FINDINGS.md, HARDWARE.md
scripts/                  setup, campaign, A/B, profiling, topology, slurm/ templates
```

`[tool.maturin]` wires `python-source = "python"` and the module name `paulistrings._paulistrings`, which `python/paulistrings/` re-exports.
`libc` is a `cfg(target_os = "linux")` target dependency used only for pinning and `set_mempolicy`, and num-complex carries `bytemuck` so exchange blocks cast coefficient columns to bytes without a copy.
