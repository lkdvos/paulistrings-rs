# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

v0.1 is **feature-complete** (`GeneralUnitary1Q/2Q` landed in v0.2 B.9; there are no `todo!()`s left). All eleven phases of `research/plans/2026-04-30-v0.1-tdd-slices.md` have landed: the full `PauliString` algebra, `PauliSum` bulk ops, the ingestion path, Clifford / rotation / noise channels, the truncation policies, the Rayon-parallel sort-merge engine, the Python bindings, and both benchmark harnesses.
`cargo test --workspace` is green (214 tests, none ignored) and `cargo clippy --workspace --all-targets -- -D warnings` is clean.
Treat `research/plans/2026-04-30-v0.1-scope.md` as the source of truth for architecture; doc comments reference its section numbers (`§3.1`, `§5`, etc.) so the mapping back to the doc is direct.

**v0.2 has landed: a GF(2)-linear bucketing engine.** Design in `research/plans/2026-08-26-v0.2-gf2-bucketing.md`, work order in `research/plans/2026-08-26-v0.2-tdd-slices.md`, measured results in `research/notes/2026-08-26-v0.2-results.md`. It **supersedes v0.1 scope §5 (sort-merge) and §9 (parallelism)**, which are marked as such in place; every other v0.1 section stands.

The short version: partition the sum by a GF(2)-linear hash `h(v) = H·v`. Since channels act on keys by `v ↦ v ⊕ d` with `d` drawn from a small set, output buckets are statically predictable — so buckets are write-disjoint, duplicate keys can never straddle a bucket, and the global sort disappears.

**v0.3 has landed (all four sections).** Plan in `research/plans/2026-08-29-v0.3-followups.md`, measured results in `research/notes/2026-08-30-v0.3-results.md`, commits `128890a`, `d1c16e4`, `ee5ac70`+`e8a885d`, `7be33f1`. What it means for anyone touching the code:

- **One type (§4).** There is a single bucketed `PauliSum`, canonically ordered `(bucket index, lex(x, z))`; a single-bucket sum — anything ≤ 1024 terms — is plain lex-sorted, which is why small fixtures see the v0.1 order. `from_sum`/`into_sum`/`to_sum`/`propagate_bucketed` are gone: `propagate` is the single entry point with no conversion cost (the v0.2 Trotter regression is erased — 137/864 ms in v0.1 → 18.8/80 ms now), `with_hash` re-partitions, `BuildAccumulator::finalize` picks the hash, iteration is `iter()`/`to_arrays()`/`bucket(b)` (the flat `x()`/`z()`/`coeff()` slice accessors no longer exist), lookup is `get(x, z)`, and `TruncationPolicy` has a single `finalize_layer`.
- **Deterministic partition (§1).** The bucket-count floor is the fixed `DEFAULT_MIN_BUCKETS = 128`; `B` was a function of the term count alone (v0.5 R1 makes it the running max — see the v0.5 block below); output remains byte-identical across machines and thread counts.
- **In-place coset layers (§2).** The engine's parallel unit is one coset of `span(h(D))` (`engine/coset.rs::Gf2Span` — the span, not `h(D)`, because a custom channel's delta set need not be XOR-closed), gathered input-major once and merged back in place via worker-persistent scratch (`begin_layer`/`end_layer`/`spare` and the `2n` double buffer are gone; peak RSS at 10⁶ terms roughly halved). A `u8` delta-index tiebreak in the per-bucket sort keeps output bitwise-identical to v0.2 and B-independent; it cost 5–14% on wide-delta layers and was deliberately kept then because dropping it breaks bitwise B-independence for channels merging ≥3 contributions per key — **removed in v0.5 S1** when that requirement was dropped by policy (see the v0.5 block below). `Channel::support()` returns a `[u64; W]` bitmask; `PauliRotation` stores no support vector. A 32-entry byte-exact **fingerprint net** (`engine/bucketed.rs::layer_fingerprints_are_stable`) pins the engine's output bits — if it goes red, analyze; never regenerate the literals casually.
- **Symmetric truncation (§3).** `TopN(n)` keeps the threshold tie group iff it fits entirely within `n` and discards it whole otherwise — it never splits an equal-magnitude group (symmetry multiplet), retains exactly `n` when magnitudes are distinct, and **deliberately wipes an all-tied sum to empty**. This changed the committed Ising CSVs (max Δ⟨X_avg⟩ 0.0049 on 4×4, 0.0012 on 6×6; three-way sensitivity table in `docs/examples/ising_2d_quench.md`) and, as a side effect, keeps truncated sums below the cap, dropping the 6×6 quench to ~11 s.

Measured (ccqlin038, 32 threads; details and caveats in the v0.3 results note): rotation layer 6.83 → ~4.8–5.1 ms, CNOT 7.06 → ~2.8–3.3 ms, `finalize_top_n` 22.6 → 10.4 ms, thread scaling ~12×, Ising 4×4+6×6 ~34.5 s → 12.3 s end to end (most of that factor is the §3 physics-rule change, not engine speed). Honest open items: `GeneralUnitary2Q` at 10⁶/32 threads is ~20–45% slower than the pre-§2 tree (rank-2 coset working set vs L2; two fixes tried and rejected by measurement), single-thread rotation is ~15% slower than v0.2 (coset overhead with no parallel payoff), and `Depolarizing` reads ~1.6 ms vs v0.2's committed 1.12 ms for reasons not yet explained (possibly machine state — the box was shared during the campaign).

**v0.4 (perf framework) and v0.5 (engine optimizations) have landed.** v0.4: the `phase-timing` feature, `examples/phase_breakdown.rs`, `scripts/{bench-campaign.sh,perf-stat.sh,profile.sh,bandwidth.sh,perf-viz.py,criterion-report.py,fit_scaling.py}`, `crates/membench`, `benchmarks/PROFILING.md`; baseline in `research/notes/2026-08-30-v0.4-perf-framework-baseline.md` (including the real bandwidth ceiling: ~39 GB/s/socket, ~45–49 both — 2 of 6 channels populated, not the 281.6 GB/s spec). v0.5 results in `research/notes/2026-08-30-v0.5-results.md`, commits `6a18fa1`, `5ed788d`, `a7cc8e9d`, `6f3493e`, `16ba505`. What it means for anyone touching the code:

- **Determinism policy (user directive, v0.5).** Floating-point associativity variation in results is accepted: bitwise B-independence is no longer required (those tests are demoted to `assert_terms_close`), the v0.3 delta tag is **gone**, and equal-key summation order across partitions is unspecified. Still bitwise and load-bearing: thread-count determinism and repeat-run determinism at a fixed configuration. The fingerprint net stays as the unintended-perturbation detector (empirically its 32 entries never moved through v0.5 — the fixtures only merge ≤2 contributions per key, where f64 addition commutes exactly).
- **Grow-only rebucket (R1/R2).** `rebucket` never shrinks (`B` = running max of `desired_bits` over the sum's history; explicit shrink via `with_hash`); refine evaluates one hash row (`Gf2Hash::row_parity`, O(n) not O(n·bits)) and refine/coarsen parallelize above 8192 terms. This erased the serial-rebucket oscillation that was 74% of gu2q and 46% of trotter wall through `propagate`.
- **Split-stream sort + fused merge (S1/S2).** `GatherRun` carries the identity-delta stream in separate pre-sorted `id` columns (keys untouched ⇒ bucket order inherited); only the rest stream is sorted (`sort_rows_with_scratch`, worker-persistent `SortScratch`, zero steady-state allocations); `merge2_into` fuses the two-stream merge with the reduction, id-first on key ties, keeping exact-zero cos rows (signed zero). **The per-run sort must stay the stable adaptive `sort_by`**: runs are concatenations of sorted/piecewise-sorted streams and the unstable sort measured +77% on rotation_zz — recorded in the fn doc; do not "simplify" it to `sort_unstable_by`.
- **Measured (cumulative v0.4 → v0.5, probe through `propagate`, n=10⁶):** gu2q 16.0× at 32t (210 → 13.1 ms/layer) and 2.45× single-thread; trotter 3.55× at 32t; rotation_zz 2.0× at 16t. Criterion (fixed B): rotation_zz −29%, rotation_w4 −31%, gu2q −36%. The three v0.3 open items are closed (rebucket oscillation; single-thread rotation now 34% *faster* than the v0.3 tree; depolarizing anomaly = machine state).
- **Negative result, recorded:** static coset→worker assignment (NUMA-affinity experiment) measured 1.25–1.9× slower than work-stealing and was reverted — see the v0.5 results note before re-attempting placement work. Trotter at 32t moves ~36 GB/s DRAM against the ~45–49 ceiling: near the memory wall, so the next wins there are traffic reduction or real NUMA partitioning, not scheduling.

The v0.1 engine is retained in `engine/sort_merge.rs` as the differential-testing oracle and as the per-layer fallback for a channel that declines to be prepared, and its `sort_phase` / `merge_into` are reused as the per-bucket kernels.

Known gaps — check these before assuming a feature works:

- `engine/sort_merge.rs` — the design doc's O(n) **bucket** phase (§5) is not implemented, and **cannot be as specified**: its claim that concatenating support-bit buckets yields a sorted array is false, because a support qubit's key bits are not the most-significant bits of the sort key. See `research/notes/2026-08-26-why-s5-concatenation-fails.md` for a four-term counterexample. `Channel::support()`, added solely to feed that phase, is dead code the engine never calls. What ships instead is a *sequential* O(n log n) comparison sort allocating a permutation plus three `Vec`s per layer — the serial bottleneck in an otherwise parallel pipeline, and the thing v0.2 removes.
- The small-sum hashmap fast path (§8.3) is unimplemented; `SMALL_SUM_THRESHOLD` is declared and never read. `#![allow(unused)]` in `lib.rs` and several modules masks this and similar dead code.
- Slice 11.3 (profile-driven optimization) never ran. Baselines now exist: raw criterion + Ising output under `benchmarks/results/2026-08-26-ccqlin038/` (gitignored) with a committed summary in `research/notes/2026-08-26-v0.1-baselines.md`. Compare against those, on that host, before claiming a speedup.
- CI runs Rust only (fmt, clippy, tests, doctests, examples, rustdoc). The PyO3 bindings and `python/paulistrings/tests/` are only ever validated locally — and `python/paulistrings/tests/test_general_unitary.py` (v0.2 B.9), `test_expectation.py`, and the v0.3 changes (numpy export via `to_arrays`, the new `TopN` docstrings) have **never been executed**: no `maturin` or venv was available on the dev host through v0.2 and v0.3. Compile-checked on the Rust side only.
- `PauliSum::from_strings` is still `pub(crate)` + `#[cfg(test)]`, so tests build sums through it or `BuildAccumulator`. (Expectation/overlap landed in v0.2 B.10 — `expectation_product_state`, `overlap`, `identity_coefficient` are public; the old note that the example hand-rolls its observable is obsolete.)

Public API change in v0.2 B.3: `PauliRotation`'s fields are now private and it is built with `PauliRotation::new(gen, theta)`, which **derives** the support from the generator. Previously `support` was caller-supplied and never validated against `gen_x`/`gen_z`; the bucketed engine reads `support()` to decide which bits to extract, so a mismatch would have been a silent miscompilation rather than a slow path. Accessors: `generator()`, `theta()`, `weight()`.

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
- **`PauliSum<const W: usize>`** — bucketed structure-of-arrays: per-bucket column triples (`Vec<[u64; W]>` for `x` and `z`, `Vec<Complex64>` for coefficients) partitioned by a GF(2)-linear `Gf2Hash`, plus `num_qubits` and a cached length. **Invariant: every term lives in `buckets[h(v)]`, strictly ascending `(x, z)` within each bucket, no duplicates; canonical order = bucket index, then key.** A single-bucket sum (≤ 1024 terms) is therefore plain lex-sorted. SoA columns keep coefficient-only and key-only scans cache-friendly and map directly to GPU device buffers.

### The central algorithm: sort-merge (not hashmap)

**This section describes the superseded v0.1 engine, which is still present as the oracle. For the shipped path see the v0.2 design doc §2 and §6.**

When a channel acts, it perturbs only the bits in its support. A sorted input therefore stays *almost sorted* — order is preserved outside the support, perturbed only inside. Each layer is a three-phase pipeline:

1. **Scan** — for each input term, channel writes ≤ `MAX_FANOUT` outputs into a pre-allocated buffer of size `n_in × MAX_FANOUT`. Embarrassingly parallel.
2. **Bucket** — *never implemented, and the claim it rested on is false* (see the gap list above and `research/notes/2026-08-26-why-s5-concatenation-fails.md`). What ships is a sequential `O(n log n)` comparison sort. The design doc's text was: partition outputs into `2^(2|support|)` buckets indexed by support bits, within each bucket relative order is inherited, so concatenation is sorted — the last clause does not follow, because support bits are not the most-significant bits of the key.
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
- The `phase-timing` Cargo feature (`engine/stats.rs`) gates a per-phase timing breakdown of the propagation engine; it is measurement-only and never in the default feature set, and its non-perturbation of engine output is checked by the existing bitwise-identity tests (the fingerprint net, thread/bucket-count/seed determinism tests) running under the feature in CI.
- The probe: `cargo run --release --features phase-timing --example phase_breakdown`.
- `scripts/bench-campaign.sh` plus `benchmarks/PROFILING.md` is the canonical change → measure → compare workflow.
- `crates/membench` + `scripts/bandwidth.sh` give the memory-bandwidth ceiling used for roofline comparisons.
