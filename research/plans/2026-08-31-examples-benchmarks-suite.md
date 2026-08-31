# Examples & benchmarks suite — adapted execution plan

## 1. Provenance & scope

This document is the repo-adapted version of an externally-written handoff ("Handoff: Examples &
Benchmarks for the Pauli Propagation Simulator (v2)") specifying a suite of example programs and
benchmarks: Part 0 shared infrastructure, Part A benchmarks A–F against `PauliPropagation.jl`, and
Part B application showcases B1–B7. The original was written **without knowledge of this
repository** and hedged its interface assumptions accordingly ("obtain the current interface stub
from the codebase owner and reconcile all names before coding"). This document *is* that
reconciliation: every interface assumption has been checked against the actual code
(`crates/paulistrings-py`, `python/paulistrings`, `crates/paulistrings`), every missing capability
is registered as a named dependency (§3), and every judgment call made in adaptation is recorded in
the decision log (§8). **This document supersedes the external handoff.** Item IDs (Part 0, A–F,
B1–B7, M0–M4) are preserved for traceability.

Scope of the current branch (`examples-benchmarks-suite`): through milestone **M2** — Part 0
infrastructure, benchmarks A/B/D/E plus the headline C, showcases B1 (1D and 2D/3D) and B2, plus
B5/B6 from M4 where they need no gated capability. M3 items (F, B3, B4) and B7 ship as design
stubs only (§3, A8). Deliverables are runnable scripts plus markdown narrative pages with
committed figures — **no notebooks** (§8, D7).

## 2. Interface mapping

The handoff's pseudo-API, rewritten against the real surface. Rule for all suite code: **never
hard-code the handoff's names; use the right column.**

| Handoff assumption | Actual API / adaptation |
|---|---|
| `truncation: {"max_weight": w, "min_abs_coeff": eps}` kwargs | Policy objects passed positionally: `paulistrings.truncation.weight(w) & paulistrings.truncation.coeff(eps)` → `PauliSum.propagate(circuit, policy, direction=...)`. Harness-level aliases `max_weight`/`min_abs_coeff` map onto these (A7); the API keeps one construction style. |
| coefficient cutoff semantics | `coeff(eps)` **drops `\|c\| <= eps` (inclusive)** — `truncation/builtin.rs:22`. Parity-relevant vs PauliPropagation.jl, whose boundary must be probed empirically (§5). Dyadic cutoffs (2⁻¹⁴…) make boundary-straddling plausible. |
| `topn`-style truncation | Exists (`truncation.topn(n)`, tie-group preserving) but **banned from all comparative runs**: jl has no equivalent, and `topn` on the right of an `\|` composition is silently inert (no `finalize_layer` under `Or`). Showcases may use it (the existing `ising_2d_quench` does). |
| "applies the adjoint to an observable internally (Heisenberg picture)" | `direction="heisenberg"` on `propagate` — reverse channel order + per-channel `apply_adjoint`, automatic. **The default is `"forward"`**, and the README's "Heisenberg picture by default" comment is stale (fix ships in Wave 6). Rule: every suite call passes `direction=` explicitly. |
| ZZ / multi-qubit Pauli rotations (`exp(iθ/2 · Z_iZ_j)`) | Not exposed to Python today; the core `PauliRotation` accepts **any** generator weight (`channel/rotation.rs:68`) — a bindings-only gap. New `gates.pauli_rotation(pauli, qubits, theta)` (A1). CNOT–RZ–CNOT decomposition is **rejected for benchmarks**: it triples layer count and shifts per-layer truncation timing, breaking term-count parity with jl, which applies ZZ natively. Convention: `pauli_rotation` implements `U = exp(-i·θ·P/2)` (the core's convention); the handoff's `exp(iπ/4·ZZ)` is `theta = -π/2`. |
| Haar SU(4) blocks, "KAK-decomposed" | Unnecessary: `gates.unitary_2q(q0, q1, matrix)` takes a raw 4×4 unitary and is first-class in the engine (`GeneralUnitary2Q::from_matrix`). Circuits pass the matrix directly; no KAK. |
| `(observable, truncation) → evolved Pauli sum`, same type in/out | Holds today: `PauliSum.propagate(...) -> PauliSum`, round-trippable (needed by B5). |
| Expectation vs. initial state, "defaulting to all-zero", product states, stabilizer-by-generators | `PauliSum.expectation(state=...)` supports exactly three **uniform** product states: `"x+"` (`\|+…+⟩`), `"y+"`, `"z+"` (`\|0…0⟩`). The default is `"x+"`, **not** all-zero — suite code always passes `state=` explicitly. Non-uniform product states = A4 (small core addition). Stabilizer-by-generators contraction = phase-2 core feature (§3); until then B7 is a design stub. `overlap()` (Hilbert–Schmidt) is the general fallback. |
| Non-unitary channel support | Present: `depolarize(p, q)`, `dephase(p, q)`, `amplitude_damping(gamma, q)` — all single-qubit. Pauli channel with independent `(px, py, pz)` = A6 (small addition). Channels on >2 qubits **panic** (`MAX_LOCAL_SUPPORT = 2`, no fallback); everything in this suite fits within 2-local. |
| Symbolic / surrogate coefficients | **Blocked.** `Complex64` is hardcoded in the `Channel` and `TruncationPolicy` trait signatures, the SoA storage, and the `Pod`/GPU-readiness constraint (`ARCHITECTURE.md §Data-Model, §GPU-Readiness`). Unbaking it is a core redesign — F/B3/B4 are design stubs pointing at the A8 analysis note. |
| Per-run statistics (peak/final term count, phase timings) | Nothing is exposed today, and **no peak-term-count exists even in Rust** (`PhaseStats.terms_in/out` are sums over layers, and the whole struct is compile-gated behind `phase-timing`). A2 adds an always-on lightweight `propagate_with_stats` (per-layer in/out + peak). Interim channel: per-layer DEBUG records on logger `paulistrings.propagate` (`"layer {k}/{n} [...]: {before} -> {after} terms, {ms} ms"`); call `paulistrings.reset_log_cache()` after level changes. **Timed runs keep logging off** — an enabled DEBUG filter adds a clock read per layer (CLAUDE.md §Performance discipline). |
| Importers: native task JSON, OpenQASM 2, Stim format | None exist (no serde anywhere in the workspace). A5 builds them as **pure-Python** translators in `python/paulistrings/interop.py`, using the `stim` and `qiskit` packages as parsers — no Rust parsing. Stim `OBSERVABLE_INCLUDE` surfaces as the observable so one `.stim` file is a complete job. Unsupported instructions **hard-error**, never silently drop. |
| Evolved-operator serialization (B5 "task file" emit) | A3: `PauliSum.from_arrays(...)` (inverse of the existing `x_array`/`z_array`/`coefficients_array` export) + a Python-level `.npz` save/load helper. |
| "prefer addressing lattice sites by coordinate" | No coordinate/qubit-label layer exists anywhere; raw `u32` indices only, and indices are currently **unchecked at the Python boundary** (silent misbehaviour/panic when out of range — A1 adds bounds checks). Coordinate↔index maps live in `examples/common/circuits.py` as plain functions with structural tests; the heavy-hex edge list is *generated*, never hand-typed (§6, Part 0). |
| Qubit counts | Python monomorphizes widths `{1,2,4,8,16}` = 64–1024 qubits; n=127 lands in W=2 (128-bit strings). Fine for everything here. |
| Single-thread pinning | `RAYON_NUM_THREADS=1` set before the first `propagate` (lazy global pool); harness verifies and asserts it (A7). Multi-thread numbers go in a separate scaling report, never the core comparison. |

## 3. Capability-dependency register

The handoff's escalation rule ("missing capabilities are named dependencies, never silently
approximated") produces this register. Statuses: **available** / **small addition** (implemented on
this branch, Wave 2) / **phase-2 core** (design stub now) / **blocked** (design stub now).

| ID | Capability | Status | Consumers |
|---|---|---|---|
| — | Clifford + 1q rotations + arbitrary 1q/2q unitaries; 3 noise channels; coeff/weight/topn policies; forward/Heisenberg propagate; round-trip; numpy export; 3 uniform product states | **available** | everything |
| A1 | `gates.pauli_rotation(pauli, qubits, theta)` — multi-qubit Pauli-string rotation binding (+ qubit bounds checks at the boundary) | **small addition** (bindings-only) | Part 0 circuits, A–C, B1, B2, B5 |
| A2 | `propagate_with_stats` — always-on per-layer (in, out) term counts + running peak; phase *timings* stay behind `phase-timing`. Merge gated on `ab-compare.sh` direction consistency (hot-loop adjacency, LTO layout sensitivity ±20–34% precedent) | **small addition** (core + bindings) | harness (0.4), all Part A |
| A3 | `PauliSum.from_arrays` + `.npz` save/load (`paulistrings/io.py`) | **small addition** | B5, report reproducibility |
| A4 | Non-uniform product states for `expectation(state=...)` (per-qubit basis spec) | **small addition** (core, moderate) | B2, D (optional), utility verification |
| A5 | Importers `circuit_from_stim` / `circuit_from_qiskit` / `circuit_from_json` (pure Python, lazy imports) | **small addition** | 0.6 data files, A (shared-file guarantee), B2 |
| A6 | Pauli channel noise `(px, py, pz)` (1q; optionally 2q depolarizing — fits `MAX_LOCAL_SUPPORT=2`) | **small addition** (core + bindings) | B2 |
| A7 | Harness truncation aliases (`min_abs_coeff`/`max_weight`) + thread-pin verification | **small addition** (pure Python) | harness (0.4) |
| A8 | Design notes: (i) symbolic-surrogate coefficient unbaking analysis; (ii) stabilizer-generator contraction (O(n²) membership test as an *expectation* feature — not stabilizer simulation, so not in conflict with the `lib.rs` non-goal) | **docs only** | F, B3, B4, B7 |
| — | Stabilizer-state input + generator-membership contraction | **phase-2 core** (stub via A8-ii) | B7 |
| — | Symbolic/surrogate coefficients, analytic gradients, landscape evaluation | **blocked** (stub via A8-i) | F, B3, B4 |

## 4. File-layout reconciliation

The handoff's proposed root tree is adopted **for Python showcases only**; benchmarks fold into the
existing `benchmarks/` conventions rather than duplicating them.

```
examples/
  common/            # circuits.py, observables.py, oracles.py, harness.py, report.py (+ __init__.py)
  data/              # checked-in .stim files (preferred), task JSON, generated heavy_hex_127.edges
                     #   + the generator script; every file carries a provenance note
  <slug>/            # one dir per showcase: README.md narrative, runnable script,
                     #   committed CSV + SVG (regenerated in the same commit as the script)
benchmarks/
  python/bench_*.py  # Part A entries, following bench_baseline.py idioms (importorskip,
                     #   seeded fixtures outside the timed region, benchmark(group=), assert-on-result)
  julia/             # Project.toml + Manifest.toml (pinned), runner.jl, README (§5)
python/paulistrings/ # interop.py (A5), io.py (A3) — shipped API, not example code
research/notes/      # A8 design notes; measured-result write-ups
```

- `tests/` for the CI-safe correctness gates lives inside `python/paulistrings/tests/` (what CI
  already runs) for API-level tests, and `examples/` scripts carry their own `test_*` collection
  run manually — CI's python job installs numpy only, so **every CI-visible test importorskips
  stim/qiskit/matplotlib**.
- Rust-side `crates/paulistrings/examples/` is untouched (it showcases the Rust API).
- `pyproject.toml`: new extra `examples = ["matplotlib", "stim", "qiskit", "qiskit-aer", "numpy"]`;
  `bench` gains `stim`.
- Handoff's `oracles_ext/` is dropped — oracle wrappers live in `examples/common/oracles.py`
  behind capability checks.
- Results land in the gitignored `benchmarks/results/<date>-<host>/` with the standard provenance
  header (commit + dirty marker, rustc, CPU, threads), append-not-overwrite. Figures that docs
  reference are committed next to their showcase.

## 5. Julia baseline appendix (PauliPropagation.jl)

There is **no Julia infrastructure in this repo** — a recorded, deliberate exclusion
(`benchmarks/python/bench_baseline.py:7-9` refuses PyJulia in pytest). Note also that
PauliStrings.jl (credited in the README as inspiration) and PauliPropagation.jl are **different
packages**; nothing here has touched either from code. The baseline is therefore built from
scratch, honoring the exclusion:

- `benchmarks/julia/Project.toml` + `Manifest.toml` pin PauliPropagation.jl (and BenchmarkTools) to
  exact versions; the recorded version goes in every results file.
- `benchmarks/julia/runner.jl` reads a **task JSON** (the shared schema frozen in the A5 design:
  n_qubits, gate list with names/qubits/params, observable as sparse Pauli dict, truncation
  `{max_weight, min_abs_coeff}`, direction, thread count) and emits JSON: expectation value, final
  and per-layer term counts, warm wall time (BenchmarkTools; first compilation run discarded).
- `benchmarks/python/julia_baseline.py` shells out via `subprocess`, **skips cleanly when no
  `julia` binary is found** (Lmod provides Julia on Flatiron hosts), and is never imported by CI.
- Semantic mapping, verified empirically before any timing is recorded:
  - Pauli-string convention: Hermitian Y on both sides (this repo: `Y → (x=1,z=1)`, no phase).
  - Coefficient-cutoff boundary: this repo drops `|c| <= eps` (inclusive); jl's comparison operator
    is probed with boundary-straddling fixtures, and any divergence is handled by the harness
    (perturbed-eps comparison) and **reported as a finding, never fudged**.
  - Truncation timing: this engine truncates after every channel (one channel = one layer). Parity
    with jl's per-gate truncation therefore requires the **one-gate-per-channel construction rule**:
    suite circuits are always built one gate per `Circuit` push. A parity smoke test (small circuit,
    layer-by-layer term counts identical, expectations agreeing to 1e-12) gates the driver before
    any timing code exists.
  - Comparative runs use only the shared knobs (`coeff`/`weight`); jl-only knobs (e.g. frequency
    truncation) and repo-only knobs (`topn`) are excluded.

## 6. Per-item adaptation blocks

Runtime classes: **CI** (runs in the numpy-only CI python job, importorskip'd extras), **manual-short**
(minutes, run while developing), **manual-long** (hours; autonomous under the time-box policy —
pilot a scaled-down run, project cost, shrink grid/lattice and record the cut if a single run
projects past ~3–4 h; workstation only, never Slurm).

### Part 0 — shared infrastructure

- **0.1 `circuits.py`** — heavy-hex kicked Ising: 127q edge list **generated from qiskit's
  `FakeSherbrooke` coupling map** by a checked-in script writing `examples/data/heavy_hex_127.edges`,
  with a structural test (node/edge counts, degree distribution ≤3); sublattice edge 3-coloring for
  the ZZ layers; sub-lattices n=2…64 for scaling. ZZ layers via A1 `pauli_rotation` (θ_J fixed at
  the Clifford point), X layer via `rx`. `xxz_chain_trotter` (XX+YY+ZZ via three `pauli_rotation`s
  per bond). `random_su4_staircase(n=36, depth, seed)` via `unitary_2q` raw Haar matrices (seeded,
  recorded). `qaoa`/`hardware_efficient_ansatz` for B-series. All circuits one gate per channel (§5).
- **0.2 `observables.py`** — `Z(q)` (canonical `Z_62`), the weight-10 and weight-17 operators with
  exact supports parameterized by lattice size, sparse XXZ Hamiltonians. Built via
  `PauliSum.from_strings` (Hermitian convention).
- **0.3 `oracles.py`** — `statevector_expectation` (qiskit Aer, n ≤ ~28, importorskip);
  `stim_clifford_exact` (primary Clifford oracle; consumes the same checked-in `.stim` file the
  engine imports via A5 — the byte-identical-input guarantee); `light_cone_exact` (exact
  shallow-depth reference by causal-cone reduction — computed, not hard-coded);
  `load_published_reference` (CSV/JSON with mandatory provenance header: source, method, accuracy).
  Tsim oracle: **optional**, behind a capability check; suite runs without it.
- **0.4 `harness.py`** — warm timings (one discarded warmup both engines), propagation vs
  contraction timed separately, peak RSS (`/proc/self/status` sampling, same approach as
  `phase_breakdown.rs`), peak/final term counts via A2, absolute error vs supplied oracle;
  single-thread pinning asserted (A7); **blocking term-count parity check** (coefficients to
  ~1e-12, order-tolerant) before any cross-engine timing; time-to-fixed-accuracy driver (sweep
  truncation until |err| < ε; report truncation, term count, time per engine).
- **0.5 `report.py`** — machine-readable results JSON (CPU, versions incl. PauliPropagation.jl,
  seeds, commit) + standard plots: error-vs-runtime per engine, term-count-vs-truncation,
  time/memory-vs-size.
- **0.6 data files** — `.stim` preferred where expressible (circuit + noise +
  `OBSERVABLE_INCLUDE`); task JSON where truncation/observable structure exceeds Stim's format;
  QASM only as interchange fallback. Provenance note per file. Stim-importer gaps hard-error and
  get named here if hit.

### Part A — benchmarks

| | Setup / adaptation | Oracle | Runtime class | Acceptance gate |
|---|---|---|---|---|
| **A** Clifford gate | Heavy-hex n=127, 5 Trotter steps, θ_h ∈ {π/2, 0}; weight-10/17 observables; evolved operator stays a single string | `stim_clifford_exact` from the shared `.stim` file; Clifford-point integers (+1, −1) asserted to machine precision at *any* truncation | **CI** (importorskip stim) + bench entry | Exact ±1; single-term evolved operator; a failure diagnostic names the likely convention bug (direction ordering / Clifford-boundary angle) |
| **B** θ_h sweep | n=127, 5 steps, θ_h ∈ {0, 0.2, π/8, π/4, 3π/8, π/2}; Z_62 + weight-10/17 | `light_cone_exact` at every θ_h; endpoints must reproduce A's integers | manual-short | Monotone convergence to exact as truncation loosens; endpoints exact; term-count parity at matched truncation |
| **C** deep Trotter (headline) | n=127, up to 20 steps, θ_h ≈ 0.6–1.0; `coeff` ∈ {2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸} (+ weight cap per convergence study) | **Self-converged reference with documented convergence evidence** (tightening-truncation stability); published values only if obtainable with clean provenance — never fabricated | **manual-long** (time-boxed; pilot first) | Reproduces reference within 0.01; term count inside the ~1.2M–9.3M envelope at matched cutoff (outside ⇒ semantics investigation, no timings reported); parity holds; error-vs-runtime plot |
| **D** XXZ chain | n = 20…100, Jz ∈ {0, 0.5}; central Z + a weight-2 term | Statevector n ≤ 26; **analytic quadratic term-growth law at Jz=0**; TDVP baseline optional (skipped if no package — named follow-up) | manual-short | Matches statevector at small n; Jz=0 growth matches theory; time/memory-vs-n for both engines |
| **E** SU(4) brickwork | n=36 staircase, fixed seed, checked-in circuit; smaller n for validation | Statevector n ≲ 28 | manual-short | Matches statevector where checkable; deterministic given seed; term-explosion-vs-depth plot |
| **F** surrogate landscape | — | — | **design stub** (blocked, A8-i) | Stub section names the dependency and the B-derived exact curve it would validate against |

### Part B — showcases

| | Adaptation | Validation | Runtime class |
|---|---|---|---|
| **B1** scrambling/OTOC | 1D first (support growth, butterfly velocity, OTOC read off the evolved sum); then 2D quench (magnetization/correlations), 3D if it scales. **Mandatory truncation-convergence panel on every curve** — these circuits violate locally-scrambling assumptions, convergence is shown, never assumed | Small-lattice statevector cross-check | 1D manual-short; 2D/3D **manual-long** (time-boxed) |
| **B2** noisy + utility verification | Per-gate noise (A6 Pauli channel + existing channels) on the 127q kicked Ising: noise-accelerates-truncation demo (term count & coefficient decay vs p); verification demo reproduces C's converged answer, noiseless limit recovers C | Reuses C's reference; noiseless-limit check | manual-short (noise shrinks the tracked set) |
| **B3** variational pre-training | **design stub** (blocked, A8-i); the estimator-vs-landscape caveat text is preserved in the stub for the future implementation | — | stub |
| **B4** QML/QCNN | **design stub** (blocked, A8-i) | — | stub |
| **B5** operator backpropagation | Propagate observable through final k layers (`direction="heisenberg"`), serialize evolved observable + residual circuit via A3/A5 task files; compose and check | Composed expectation = full-circuit expectation within truncation error | manual-short |
| **B6** resource probes | Stabilizer-Rényi-style magic proxies and operator-entanglement across depth/size, computed in pure Python over `x_array`/`z_array`/`coefficients_array` — read-only, no core changes | Small-system exact comparison | manual-short |
| **B7** stabilizer-prep → PP-estimate | **design stub** (phase-2 core, A8-ii): generator-membership contraction is an expectation feature (not stabilizer simulation), but a real core addition; never approximated by a 2ⁿ-term expansion | — | stub |

## 7. Milestones & global rules

- **M0** — Part 0 + Benchmark A (CI-green gate; Stim oracle + Stim-import path end-to-end).
- **M1** — Benchmarks B, D, E + Showcase B1-1D.
- **M2** — Benchmark C (headline) + Showcase B2 + B1-2D/3D.
- **M3** *(deferred)* — F, B3, B4: design stubs on this branch (A8-i).
- **M4** — B5, B6 implemented; B7 stubbed (A8-ii).

Global rules (restated from the handoff, all binding):
1. Every numeric claim is computed by an oracle or loaded from a provenance-tagged reference file.
   Only Clifford-point integers and oracle outputs may be asserted directly. **No fabricated
   reference values.**
2. **Term-count parity blocks timing**: no cross-engine timing is reported for a run whose evolved
   sums diverge term-for-term (coefficients ~1e-12, order-tolerant) at matched truncation.
3. Single-threaded by default; CPU model / library versions / seeds recorded; multithread numbers
   in a separate, labeled scaling report.
4. Every real-time-dynamics or truncated result ships with a convergence panel.
5. Missing capabilities are escalated as named dependencies (§3), never silently approximated.
6. Each example is reproducible from its checked-in data files; where a `.stim` file exists it
   drives both the engine and the oracle.
7. (Repo-specific) Warm-timing discipline both sides; `RUST_LOG` unset in timed runs; noise floor
   is ±5–8% single-threaded on the reference host, so any cross-engine claim under ~10% needs the
   `ab-compare.sh` direction-consistency protocol, not a single campaign.

## 8. Decision log

Judgment calls made in this adaptation, for post-hoc review:

- **D1 — Python-level importers over Rust parsers.** `stim` and `qiskit` are already optional deps
  and battle-tested parsers; a Rust Stim/QASM parser would add serde + a format surface to the
  core for zero engine benefit. `interop.py` translates parsed circuits into `Circuit` calls.
- **D2 — SU(4) as raw matrix, no KAK.** `unitary_2q` makes decomposition pure overhead and a
  needless parity hazard (extra layers shift truncation timing).
- **D3 — `topn` banned from comparative runs** (no jl equivalent; `Or`-composition inertness sharp
  edge); allowed in showcases.
- **D4 — ZZ via a new `pauli_rotation` binding, not CNOT–RZ–CNOT** (parity + 3× layer count; the
  core already supports any generator weight — bindings-only change).
- **D5 — Benchmark C reference is self-converged with documented convergence evidence**; published
  values only with clean provenance. Avoids fabrication risk while keeping the <0.01 bar.
- **D6 — Julia driver is subprocess + pinned Project.toml, never pytest/PyJulia**, honoring the
  recorded exclusion; skips cleanly when Julia is absent; out of CI.
- **D7 — Scripts + markdown pages over notebooks.** The repo has zero notebook infra and an
  established script + doctested-walkthrough + committed-figure pattern; notebooks would add CI and
  diff cost without fitting any existing convention.
- **D8 — Root `examples/` for Python showcases only**; Part A folds into `benchmarks/python/`
  idioms; Rust examples untouched.
- **D9 — Explicit `direction=` and `state=` everywhere** (stale README default claim; `"x+"`
  expectation default ≠ handoff's all-zero assumption). README fix ships in Wave 6.
- **D10 — One-gate-per-channel construction rule** for all suite circuits, making per-layer
  truncation timing comparable with jl's per-gate truncation.
- **D11 — Stats split**: term counts become always-on and Python-visible (A2); phase *timings*
  remain behind the `phase-timing` compile feature. A2 merges only behind an ab-compare
  direction-consistency pass (hot-loop/LTO sensitivity precedent).
- **D12 — B6 computed in pure Python over the numpy export** (read-only diagnostics; no core
  additions for a niche showcase).
- **D13 — B7 reframed as an expectation feature** (generator-membership contraction), so it does
  not conflict with the `lib.rs` stabilizer-simulation non-goal; still phase-2 core work, stubbed.
- **D14 — `oracles_ext/` dropped**; capability-gated wrappers live in `examples/common/oracles.py`.
- **D15 — Time-box policy for manual-long runs** (pilot → project → shrink grid/lattice with the
  cut recorded) instead of scheduled user check-ins; Slurm is never invoked.
