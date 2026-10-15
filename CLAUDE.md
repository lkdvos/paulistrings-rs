# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

`paulistrings-rs` implements Pauli propagation: operators evolved in the Pauli basis under gates and noise channels, with truncation keeping the sum tractable.
It is a Rust core (`PauliSum<W>`, a bucketed structure-of-arrays sum) with PyO3 bindings, and optional partitioned runs over NUMA domains, MPI ranks (`mpi`) or CUDA devices (`cuda`).
Correctness contracts live in the docs of the item that enforces them; `research/FINDINGS.md` (measured and rejected ideas) and `research/HARDWARE.md` (host facts) are cited by heading, so sweep citations when renaming one.

## Comment & prose style

These rules are binding on every file, including markdown.

1. **One sentence per line.** Never reflow a paragraph across lines. A long sentence stays one long line.
2. **Comment blocks are rare.** A `//!` module header is at most five lines and says what the module contains, not how it works, unless it states the module's contracts, which may run longer. A run longer than ~8 comment lines needs a reason.
3. **No history in comments.** No dates, no "we tried X", no "previously", no A/B tables, no "re-measured on <host>". That is what `research/FINDINGS.md` and git are for.
4. **Cite, don't paraphrase.** Cite a FINDINGS heading or the item that holds the contract; one line outranks the paragraph restating it.
5. **Comment the surprise** — a safety invariant, a non-obvious algorithm choice, a contract an implementor must honour. Never the mechanics the code already states.
6. **Rust docs are a developer guide.** Python docstrings and the book are the user guide, so rustdoc carries no usage examples or doctests, and multi-paragraph `///` belongs only on the `pub` surface.

## Code organisation

- Dependencies point downward only.
- Modules are private; the user API is re-exported flat at the crate root, `gpu` and `mpi` are public modules behind their features, everything else is `pub(crate)`.
- About 800 production lines per file is a soft cap; split on real seams.
- No abbreviations in names (`accumulator`, not `acc`); domain acronyms (GF2, PTM, NUMA, MPI, NCCL) stay.
- Moving an item sweeps the citations of it in the same commit.

## Commands

Python tooling is uv; the Rust toolchain is pinned in `rust-toolchain.toml`.
`uv run` rebuilds the editable extension (release mode, written into `python/paulistrings/`) whenever Rust sources or build config change, so prefer it to a bare `python`/`pytest`.
On Flatiron `module load uv`; a first `uv sync` needs uv ≥ 0.8.
Point `UV_CACHE_DIR` at local disk (`/home/$USER/.cache/uv`, `UV_LINK_MODE=copy`) to keep the cache off the `/mnt/home` inode quota.

```bash
scripts/setup.sh                                                 # once: uv sync plus the examples extra
cargo test --workspace                                           # must be green at every commit
cargo test -p paulistrings -- --ignored                          # the opt-in exhaustive tier
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest python/paulistrings/tests
MATURIN_PEP517_ARGS="--profile dev" uv run pytest ...            # debug build; every uv command on that venv must then set it
```

Benchmark `--release` only (`lto = "fat"`, `codegen-units = 1`).
After changing a script, the probe or a Slurm template, `scripts/smoke-tooling.sh` runs every measurement tool once at smoke size and skips what the host lacks.
The measurement probe, which also drives the partitioned engine (`--partition-cpus` takes one cpulist per partition, as `numactl -H` prints them):

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
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 cuda/12.8.0 nccl/2.23.4-1 python-mpi/3.12.9 uv
export LIBCLANG_PATH=$(llvm-config --libdir)
cargo test -p paulistrings --features cuda,mpi,test-utils --test mpi_ranks
scripts/mpi-test.sh --ranks 2,4 --cuda [--python]
cargo clippy -p paulistrings-py --features cuda,mpi -- -D warnings
```

Quiet-box campaigns run on an exclusive Slurm node from `scripts/slurm/`.
**Submitting is the user's step, never an agent's** — adjust the template and hand over the `sbatch` line.

## Releasing

Rust and Python share one version: `scripts/bump-version.sh X.Y.Z` bumps `Cargo.toml` and `pyproject.toml` together, committed together (CI's `version-sync` job enforces it).
Before tagging: `cargo test --workspace`, clippy, and the `mpi`, `python` and `docsrs` CI jobs green on `main`; the `docsrs` job is what docs.rs builds, locally `RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +nightly doc --no-deps -p paulistrings --features phase-timing,test-utils,cuda`.
Dry-run `.github/workflows/release.yml` via `workflow_dispatch` before the real tag.
**Pushing the tag is the user's step, never an agent's**: `git tag vX.Y.Z && git push origin vX.Y.Z` triggers `release.yml`, which builds the wheels (`cuda` on Linux and Windows, never `mpi`) and the sdist, attaches them to the GitHub Release, and publishes them to PyPI by Trusted Publishing from the `pypi` environment.
crates.io publishing is a separate manual `workflow_dispatch` of `.github/workflows/crates-publish.yml` (default `--dry-run`), after the wheel release; it authenticates by crates.io Trusted Publishing, so the repository holds no registry token.

## Progress logging

Log targets are constants next to their modules; Python reaches them via `pyo3-log`, and `paulistrings.reset_log_cache()` must follow a mid-process level change.

## Testing & TDD policy

Development is test-driven: red (the smallest failing test that pins the behavior), green (the minimum to pass), refactor (only after green).

- Unit tests live in a sibling `<module>/tests.rs`; the module file keeps one `#[cfg(test)] mod tests;` line, so tests keep private access.
- Cross-module and integration tests live in `crates/paulistrings/tests/`; shared fixtures live in `test_support`, so add helpers there rather than copy-pasting.
- Tests assert hand-computed expected values; where a reference identity exists (`XZ = -iY`), encode it.
- Parameterize multi-qubit and multi-word logic over `W ∈ {1, 2}`; property tests (`proptest`) cover the algebraic laws.
- No `#[ignore]`d tests outside the opt-in exhaustive tier, `#[ignore = "exhaustive: <reason>"]`, which never substitutes for a default-run test of the same behaviour.
- Every device test opens with `test_support::require_cuda!()`.
- Commit logical units and check in with the user at feature boundaries.

## Determinism policy

Correctness is agreement to floating-point tolerance: `test_support::assert_terms_close`, and `test_support::assert_runs_agree` for runs whose truncation boundaries differ.
Output bits are not a property of the library, so never design, constrain or reject anything to keep them stable.

## Performance discipline

- Every cargo feature is off by default.
- Benchmark `--release` only, keep input generation outside the timed region, and report single- and multi-thread numbers separately.
- Effects below the noise floor need `scripts/ab-compare.sh`, whose criterion is direction consistency across every pair, with the median Δ% as the effect size; a direction-consistent phase delta is not an effect if the total is flat.
- Run campaigns with `RUST_LOG` unset; every benchmarking script sources `scripts/jcc-rustflags.sh`.
- Record SM and memory clocks (`nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv`) with every GPU number.

**Read `research/FINDINGS.md` before re-attempting an optimization idea.**

Open limitations are GitHub issues.

## Repo layout

```
crates/       paulistrings (Rust core), paulistrings-py (PyO3 bindings), membench (bandwidth probe)
python/       the shipped `paulistrings` package and its tests
benchmarks/   benchmark suites and PROFILING.md
examples/     Python showcase suite
docs/book/    mdBook site, canonical for user-facing prose
research/     FINDINGS.md, HARDWARE.md
scripts/      setup, campaign, A/B, profiling, release and slurm/ templates
```
