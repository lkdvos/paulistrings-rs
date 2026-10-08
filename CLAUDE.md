# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

`paulistrings-rs` implements Pauli propagation: evolving operators in the Pauli basis under gates and noise channels, forward or in the Heisenberg picture, with truncation keeping the sum tractable.
The one storage type is a bucketed `PauliSum<W>`: structure-of-arrays `x`/`z`/coefficient columns partitioned by a GF(2)-linear hash, so a layer's work splits into write-disjoint cosets with no global sort and no synchronization inside a layer.
`W` is a const generic; the PyO3 bindings monomorphize `W ∈ {1, 2, 4, 8, 16}` (64–1024 qubits).
The optional partitioned engine splits the sum over `P ≤ 64` NUMA domains, MPI ranks (`mpi` feature) or CUDA devices (`cuda` feature) by GF(2) partition rows, behind a sealed `Transport` seam.

`ARCHITECTURE.md` is the design source of truth; code cites its sections as `ARCHITECTURE.md §Engine`, so never rename its `##` headings without sweeping the citations.
`research/FINDINGS.md` records measured and rejected ideas, `research/HARDWARE.md` the measured host facts; both are cited by heading the same way.

## Comment & prose style

These rules are binding on every file, including markdown.

1. **One sentence per line.** Never reflow a paragraph across lines. A long sentence stays one long line.
2. **Comment blocks are rare.** A `//!` module header is at most five lines and says what the module contains, not how it works. A run longer than ~8 comment lines needs a reason.
3. **No history in comments.** No dates, no "we tried X", no "previously", no A/B tables, no "re-measured on <host>". That is what `research/FINDINGS.md` and git are for.
4. **Cite, don't paraphrase.** `ARCHITECTURE.md §Engine` is one line and outranks the paragraph restating it.
5. **Comment the surprise** — a safety invariant, a non-obvious algorithm choice, a contract an implementor must honour. Never the mechanics the code already states.
6. **Docs on private items are terse.** Multi-paragraph `///` belongs only on the `pub` surface that rustdoc and Python users see.

## Code organisation

- Dependencies point downward only: leaf algebra (`pauli_string`, `phase`, `rng`) → `pauli_sum` → `channel`/`circuit` → `truncation` → `engine` → `engine/partitioned` → `engine/gpu`; `readout` reads `pauli_sum`, never the other way.
- Modules are private; the user API is re-exported flat at the crate root, `gpu` and `mpi` are public modules behind their features, everything else is `pub(crate)`.
- Each entry point is one plain call plus one `_with(…, options)` variant (`propagate`/`propagate_with`, `scatter`/`scatter_with`).
- About 800 production lines per file is a soft cap; split on real seams.
- No abbreviations in names (`accumulator`, not `acc`); domain acronyms (GF2, PTM, NUMA, MPI, NCCL) stay.
- A move is its own commit and sweeps `ARCHITECTURE.md` citations and the layout below in the same commit.

## Commands

Python tooling is uv; the Rust toolchain is pinned in `rust-toolchain.toml`.
`uv run` rebuilds the editable extension (release mode, written into `python/paulistrings/`) whenever Rust sources or build config change, so prefer it to a bare `python`/`pytest`.
On Flatiron `module load uv`; a first `uv sync` needs uv ≥ 0.8.
Point `UV_CACHE_DIR` at local disk (`/home/$USER/.cache/uv`, `UV_LINK_MODE=copy`) to keep the cache off the `/mnt/home` inode quota.

```bash
scripts/setup.sh                                                 # once: uv sync plus the examples extra
cargo test --workspace                                           # must be green at every commit
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest python/paulistrings/tests
MATURIN_PEP517_ARGS="--profile dev" uv run pytest ...            # debug build; every uv command on that venv must then set it
```

Benchmark `--release` only (`lto = "fat"`, `codegen-units = 1`).
The measurement probe, which also drives the partitioned engine (`--partition-cpus` from `scripts/host-topology.sh`):

```bash
cargo run --release --features phase-timing --example phase_breakdown -- \
  --partitions 2 --partition-cpus "0-7,16-23;8-15,24-31" --threads 32 --n 1000000 --layers su4
```

MPI (`mpi` feature) needs the modules loaded and `libclang` for rsmpi's bindgen; it is never in a released wheel, so the bindings build into their own non-editable `./.venv-mpi`, and `uv run` must never target that venv.

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 python-mpi/3.12.9 uv
export LIBCLANG_PATH=$(llvm-config --libdir)
cargo test -p paulistrings --features mpi                        # tests/mpi_ranks.rs as a one-rank world
scripts/mpi-test.sh --ranks 2,4 [--release] [--python]           # under mpirun; --python syncs .venv-mpi
scripts/sync-mpi-venv.sh [--cuda] [--debug]                      # build .venv-mpi alone
```

Both crates' `build.rs` exist only for `mpi` (the cdylib needs its own link arg to find `libmpi.so.40`).

CUDA (`cuda` feature) needs no toolkit to compile; `cudarc` loads `libcuda`/`libnvrtc` at runtime, and NVRTC output is cached in `$PAULISTRINGS_KERNEL_CACHE` (default `~/.cache/paulistrings/kernels`, `off` to disable).

```bash
module load cuda/12.8.0
cargo test -p paulistrings --features cuda                       # passes without a device
cargo clippy -p paulistrings-py --features cuda -- -D warnings
MATURIN_PEP517_ARGS="--features cuda" uv run pytest python/paulistrings/tests/test_cuda.py
```

One GPU per MPI rank (`gpu::MpiGpuSum`) needs both features plus NCCL (`dlopen`ed), and a device per rank above one:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 cuda/12.8.0 nccl/2.23.4-1
export LIBCLANG_PATH=$(llvm-config --libdir)
cargo test -p paulistrings --features cuda,mpi,test-utils --test mpi_ranks
scripts/mpi-test.sh --ranks 2,4 --cuda [--python]
cargo clippy -p paulistrings-py --features cuda,mpi -- -D warnings
```

Quiet-box campaigns run on an exclusive Slurm node from `scripts/slurm/`.
**Submitting is the user's step, never an agent's** — adjust the template and hand over the `sbatch` line.

## Releasing

Rust and Python share one version: `scripts/bump-version.sh X.Y.Z` bumps `Cargo.toml` and `pyproject.toml` together, committed together (CI's `version-sync` job enforces it).
Before tagging: `cargo test --workspace`, clippy, and the `mpi` and `python` CI jobs green on `main`, plus `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p paulistrings --features phase-timing,test-utils` (what docs.rs builds).
Dry-run `.github/workflows/release.yml` via `workflow_dispatch` before the real tag.
**Pushing the tag is the user's step, never an agent's**: `git tag vX.Y.Z && git push origin vX.Y.Z` triggers `release.yml`, which publishes wheels to the GitHub Release.
crates.io publishing is a separate manual `workflow_dispatch` of `.github/workflows/crates-publish.yml` (default `--dry-run`), after the wheel release.

## Progress logging

The library logs through `log` under target `paulistrings::propagate`: INFO per `propagate` call, DEBUG per layer (`RUST_LOG=paulistrings=debug` with `env_logger`).
From Python it reaches `logging` via `pyo3-log`; call `paulistrings.reset_log_cache()` after changing levels mid-process.

## Testing & TDD policy

Development is test-driven: red (the smallest failing test that pins the behavior), green (the minimum to pass), refactor (only after green).

- Unit tests live in a sibling `<module>/tests.rs`; the module file keeps one `#[cfg(test)] mod tests;` line, so tests keep private access.
- Cross-module and integration tests live in `crates/paulistrings/tests/`.
- Shared fixtures live in `test_support` (`#[doc(hidden)]`, behind `test-utils`), which also exposes the crate-private pieces tests, benches and the probe need: `InProcessTransport`, `Prepared`, `apply_layer_bucketed`, the bucket constants (`DEFAULT_MIN_BUCKETS`, `DEFAULT_TARGET_BUCKET_LEN`) and `TIMER_READ_OVERHEAD_NS`; add helpers there rather than copy-pasting.
- Tests assert hand-computed expected values; where a reference identity exists (`XZ = -iY`), encode it.
- Parameterize multi-qubit and multi-word logic over `W ∈ {1, 2}`.
- Property tests (`proptest`) cover the algebraic laws: multiplication associativity, sortedness/uniqueness after a merge, idempotence of `truncate(0.0)`.
- The engine's differential oracle is `test_support::naive_apply_layer`, a direct `Channel::apply` loop sharing no code with the bucketed path.
- The partitioned engine's nets are `tests/propagate_partitioned.rs` and `engine/partitioned/layer/tests.rs`, all under `Placement::Unpinned`.
- The distributed nets are `tests/propagate_distributed.rs` (`InProcessTransport`, default `cargo test`) and `tests/mpi_ranks.rs` (`MpiTransport` under `mpirun`, `harness = false`, collective cases in one order on every rank).
- The CUDA nets are `tests/propagate_gpu.rs` and `tests/propagate_gpu_partitioned.rs` (every device test opens with `test_support::require_cuda!()`) and `python/paulistrings/tests/test_cuda.py` (skips unless `cuda_available()`); `PAULISTRINGS_GPU_TEST_DEVICES` (csv of ordinals, default `0`) points the partitioned net at real devices.
- No `#[ignore]`d tests; benchmarks follow tests.
- Commit logical units and check in with the user at feature boundaries.

## Determinism policy

Bitwise output preservation is **not required — anywhere, for anything**.
The correctness bar is agreement to floating-point tolerance (`test_support::assert_terms_close`); equal-key summation order is unspecified.
A test pinning exact bits is a tripwire: when it trips under a change that is correct to tolerance, regenerate its literals or demote it to `assert_terms_close` in the same commit.
Never design, constrain or reject an optimization to keep output bits stable.
`propagate_partitioned` at `P = 1` is byte-identical to `propagate`; across partition counts the bar is tolerance; a device run is bitwise reproducible on one device.

## Performance discipline

- Every cargo feature is off by default, and the default build must be byte- and performance-identical to one without the feature (`build.rs` emits nothing without `CARGO_FEATURE_MPI`).
- Benchmark `--release` only, keep input generation outside the timed region, and report single- and multi-thread numbers separately.
- `scripts/bench-campaign.sh` plus `benchmarks/PROFILING.md` is the change → measure → compare loop; run campaigns with `RUST_LOG` unset.
- Effects below the noise floor (`research/HARDWARE.md §Measurement noise`) need `scripts/ab-compare.sh`, whose criterion is direction consistency across every pair, with the median Δ% as the effect size.
- A direction-consistent phase delta is not an effect if the total is flat; let instruction count settle it.
- Re-derive, never inherit, any constant tuned by wall-clock A/B on an unpadded build (`research/FINDINGS.md §Rule: suspect any constant tuned by wall-clock A/B before 2026-09-10`).
- A partitioned cell (`P > 1`) runs under no `numactl`/`taskset` prefix; the engine's own pinning is the only one in effect. P=1 vs P=2 is a runtime-knob A/B: `scripts/ab-compare.sh --probe-b '<args with --partitions 2>'`.
- The probe's sub-phases (`append_ns`, `chunk_wait_ns`) are contained in their parent phase, never summed into a total; contract (a) in `benchmarks/PROFILING.md` is what to update when `phase_breakdown.rs::json_line` or `PhaseStats` changes.
- Every benchmarking script must source `scripts/jcc-rustflags.sh` (JCC erratum padding, opt-in per CPU); an exported `RUSTFLAGS` replaces `.cargo/config.toml`'s list wholesale.
- Roofline denominators come from `crates/membench` via `scripts/bandwidth.sh` (`--device` for a GPU) against `research/HARDWARE.md`.
- Record SM and memory clocks (`nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv`) with every GPU number.
- A GPU timing is the second application of a gate on the saturated sum; its dense CPU reference is `(T₃ − T₁)/2` over `--reps 3` and `--reps 1`, never `wall/3`.
- `PAULISTRINGS_GPU_EXCHANGE_BYTES` is the device layer's one runtime knob; the sender-side merge and the Clifford scatter path are `GpuLayerOptions` fields, so an A/B of either is a code A/B.

**Read `research/FINDINGS.md` before re-attempting an optimization idea.**

## Known gaps

- Pauli strings are Hermitian everywhere they are parsed: `Y` is `(x=1, z=1)` with no phase; phases arise only from products (`mul_assign` returns `i^k`).
- `PauliSum::from_strings` is test-only (`pauli_sum/tests.rs`); Rust tests otherwise build sums through `BuildAccumulator`.
- A channel with support above `MAX_LOCAL_SUPPORT = 2` makes `propagate` panic; only `PauliRotation` is exempt.
- Partitioned mode rejects exact `TopN`; `ApproxTopN` is partition-exact and the partitioned default. On devices exact `TopN` runs only at one partition.
- Device drivers take a `BuiltinTruncation`, so a custom `TruncationPolicy` cannot run on a device.
- From Python, multi-device and `comm=` with `device=` runs scatter and gather on every call; only the one-device `GpuPauliSum` stays resident.
- Thread and memory pinning are Linux-only.
- A distributed run is one partition per rank, power-of-two rank count, input replicated on every rank, raw host bytes on the wire (same architecture and `W` everywhere).
- Partition rows are random by default; `partition_row_blocks=` and `partition_row_exclude=` are the manual levers, and automatic row choice is open (`research/FINDINGS.md §Partition rows without a known lattice`).
- `CollapseSample` counts collapses on rank 0 only in partitioned and MPI runs, and has no device form (`GpuError::Unsupported`, `NotImplementedError` from `device=`).
- `DistributedSum::rotated_overlap` panics unless partition rows avoid the flipped coordinates.
- The debug test binary overflows its stack about 1 run in 4 under full parallelism; `.cargo/config.toml` sets `RUST_MIN_STACK = "16777216"`, root cause open.
- A device partition holds one export volume and one receive volume during a remote layer; `exchange_bytes` chunks the receive only.
- An MPI device group above one rank needs NCCL ≥ 2.22 and one device per rank; the chunked receive is untested over multi-rank NCCL, and NCCL across nodes is untested.
- A device partition in a group cannot refine off-schedule: an oversize block or segment is `Unsupported`.
- Peer access to pooled allocations needs the memory-pool grant in `try_enable_peer_access` (`engine/gpu/wire/peer.rs`).

## Repo layout

```
crates/paulistrings/      pure Rust core, no Python deps
  src/                    pauli_string, phase, rng, circuit,
                          pauli_sum/{storage,partition,hash,accumulator},
                          channel/{clifford,rotation,unitary,noise,identity,prepared},
                          truncation/{builtin,tree},
                          engine/{bucketed,coset,merge,direct,stats,cuda_context},
                          engine/partitioned/{backend,distributed,driver,export,layer,mpi,plan,
                          rows,runtime,sum,topology,trace,transport,truncation},
                          engine/gpu/* (behind `cuda`; `nccl` behind `cuda` and `mpi`),
                          readout/{product_state,stabilizer,echo}, examples, test_support
  tests/ benches/ examples/ docs/examples/
crates/paulistrings-py/   PyO3 bindings, cdylib `_paulistrings`, abi3-py39
crates/membench/          STREAM-style bandwidth probe behind scripts/bandwidth.sh
python/paulistrings/      the shipped package: extension re-export, interop.py, io.py, tests/
benchmarks/               python/ suites, julia/ baseline, PROFILING.md, gitignored results/
examples/                 Python showcase suite; common/ is the shared library
docs/book/                mdBook site, published by CI; canonical for user-facing prose
research/                 FINDINGS.md, HARDWARE.md
scripts/                  setup, campaign, A/B, profiling, topology, slurm/ templates
```
