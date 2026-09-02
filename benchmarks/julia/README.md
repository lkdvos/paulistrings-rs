# PauliPropagation.jl baseline

Out-of-CI, subprocess-driven baseline for cross-engine comparisons against **PauliPropagation.jl**.
Nothing here is imported by the Python package or by CI; the only entry points are a Julia script and
a `subprocess` wrapper.

| file | role |
|---|---|
| `Project.toml`, `Manifest.toml` | pinned environment: PauliPropagation.jl 0.8.2, JSON3 1.14.3, BenchmarkTools 1.8.0, resolved for julia 1.12.6 |
| `runner.jl` | reads a task JSON (schema v1), propagates, emits a result JSON |
| `probes.jl` | the semantics probes whose output every claim below is taken from |
| `../python/julia_baseline.py` | `subprocess` wrapper: builds/validates the task JSON, invokes the runner, parses the result, skips cleanly with no `julia` |
| `../python/test_julia_parity.py` | the parity gate — blocks cross-engine timing |

```bash
julia --project=benchmarks/julia benchmarks/julia/runner.jl task.json          # -> stdout JSON
julia --project=benchmarks/julia benchmarks/julia/probes.jl                    # semantics probes
python benchmarks/python/julia_baseline.py --self-test                         # wrapper smoke
python benchmarks/python/test_julia_parity.py                                  # parity report
pytest benchmarks/python/test_julia_parity.py -q                               # parity gate
```

The first Julia run precompiles (~30 s at first use, then cached). `julia` is found on `PATH`, at
`$JULIA_BINARY`, or at `~/.juliaup/bin/julia`; Lmod also provides one (`module load julia`).

There is no PyJulia / juliacall anywhere: `bench_baseline.py` records the same exclusion for
`PauliStrings.jl`. Subprocess only.

## Interchange format

`runner.jl` and `paulistrings.interop.load_task` both implement task-JSON schema v1
verbatim: same keys, same gate-name vocabulary, Hermitian-Y convention. Unknown top-level keys, unknown gate names, unknown gate fields,
and missing required keys are hard errors on both sides; the schema is versioned instead of tolerant.

Two things the runner does not do, on purpose:

* `circuit.stim_file` is a hard error — PauliPropagation.jl has no Stim parser, so a Stim-sourced
  circuit must be expanded into an inline `gates` list on the Python side.
* `run.direction` is never defaulted. jl's own default is `heisenberg=true`, this repo's is
  `"forward"`; defaulting either way would silently pick a picture.

One gate object = one channel on both engines (see P5 below).

### Non-schema runner knobs

Environment variables, so the frozen schema stays frozen. They change how the run is executed, never
what it means.

| variable | default | effect |
|---|---|---|
| `PP_BACKEND` | `dict` | `dict` = jl's `PauliSum` (hash map); `vector` = `VectorPauliSum` (the array/threaded backend) |
| `PP_WARM_REPEATS` | `3` | timed warm propagations after the cold one; `wall_warm_s` is the minimum |
| `PP_LAYER_COUNTS` | `1` | collect per-gate term counts (`@countpaulis`) in an extra, untimed propagation |
| `PP_FUSED` | `0` | experimental fused rotation kernel (`src/Performance/fused_dict.jl`); truncates during gate application, so term-count parity is not established for it |
| `PP_EMIT_TERMS` | `0` | also emit the evolved sum term-by-term when it has at most N terms (parity debugging) |

Julia's worker count is a command-line flag, not an environment variable, so `julia_baseline.py`
passes `-t{run.threads}`; the runner warns to stderr if `Threads.nthreads()` disagrees with the task.

## Parity gate result

`pytest benchmarks/python/test_julia_parity.py` — 32 tests, all passing on ccqlin038 (julia 1.12.6,
PauliPropagation 0.8.2, `RAYON_NUM_THREADS=1`).

The headline case is a 6-qubit, 57-gate circuit (`h` ×6, then 3 × [5 × `cnot`, 6 × `rz`, 6 × `rx`]),
observable `1.0·Z₂ + 0.5·(Z₁Z₄)`, one gate per channel:

| direction | state | `min_abs_coeff` | `max_weight` | layers compared | final terms (rust / jl) | \|Δexpectation\| |
|---|---|---|---|---|---|---|
| heisenberg | z+ | 1e-4 | — | 57 | 3813 / 3813 | 5.55e-17 |
| heisenberg | x+ | 1e-4 | — | 57 | 3813 / 3813 | 0 |
| heisenberg | z+ | 1e-6 | 4 | 57 | 236 / 236 | 0 |
| heisenberg | z+ | — | — | 57 | 3881 / 3881 | 5.55e-17 |
| forward | z+ | 1e-4 | — | 57 | 3926 / 3926 | 0 |

All 57 per-layer term counts are identical in every row, not just the final count, and expectations
agree to ≤5.6e-17 against a 1e-12 bar (`@countpaulis` and this engine's `layer {k}/{n}` DEBUG line
both record in application order, so for `direction="heisenberg"` both lists run backwards through
the task and line up index by index). The truncated rows are non-vacuous: the same circuit with no
policy keeps 3881 terms, so the 1e-4 row exercises coefficient truncation and `max_weight=4` exercises
weight truncation.

Beyond the headline circuit, every schema-v1 gate name gets its own single-gate task compared term by
term (coefficients to 1e-12, not just the contracted expectation, which is blind to a Y sign that
cancels): `h s x y z cnot cz swap rz rx ry pauli_rotation depolarize dephase amplitude_damping
pauli_channel depolarize2 unitary_1q unitary_2q`, plus reversed-qubit variants of `cnot` and
`unitary_2q` to catch a transposed index. All identical, with no exceptions.

Amplitude-damping semantics agree between engines; the parity gate covers it
(`research/notes/2026-09-01-julia-amplitude-damping-transposition.md` carries the bug record).

## Semantics probes

Output of `julia --project=benchmarks/julia benchmarks/julia/probes.jl` (PauliPropagation 0.8.2,
julia 1.12.6). Each probe's expected value is hand-derived in a comment in that file, never read back
from the library; the paulistrings-side counterparts are the tests in `../python/test_julia_parity.py`.

| # | convention | current semantics |
|---|---|---|
| P1 | qubit indexing | jl: 1-based, leftmost string char = qubit 1. This repo: 0-based, leftmost = qubit 0. Same left-to-right order — observable keys map verbatim, gate qubit indices map with `+1`. Internal 2-bit codes differ (jl `I=0 X=1 Y=2 Z=3`, this core `I=0 X=1 Z=2 Y=3`) but never cross the boundary. |
| P2 | Hermitian-Y & direction | Both Hermitian-Y: a real coefficient multiplies the literal string, `Y` carries no phase. `"heisenberg"`/`heisenberg=true` (jl's default) reverses gate order (`U† O U`); `"forward"`/`heisenberg=false` uses written order (`U O U†`). `S X S† = +Y`, `S† X S = -Y` on both engines; `test_hermitian_y_sign` pins the sign. |
| P3 | `min_abs_coeff` boundary — **disagree** | jl's `truncatemincoeff` keeps `abs(c) < min_abs_coeff` (a coefficient exactly at the threshold survives); this repo's `CoefficientThreshold` keeps `|c| > eps` (drops it). Measure-zero for generic angles; matters at dyadic cutoffs (2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸) with Clifford-point angles, where coefficients are exact dyadics and can land on the cutoff bit-exactly. Mitigation: perturb the threshold by one ulp and report it — never adjust a coefficient. `test_known_divergence_coefficient_boundary` pins it. |
| P4 | `max_weight` boundary — agree | Both keep `weight == max_weight`. No mitigation needed. |
| P5 | truncation granularity | jl has no layer concept: `_applymergetruncate!` (`applytoall!` → `merge!` → `truncate!`) runs once per gate. This engine truncates once per channel, so the two coincide iff one gate object is one channel — the schema's structural rule. Both record term counts in application order, so `heisenberg=true`/`"heisenberg"` counts run backwards through the circuit as written on both sides, comparing index-by-index with no reversal. |
| P6 | noise parameter mapping | jl's `PauliNoise` channels damp by `1-λ`; this repo takes a probability `p`. `depolarize(p)` → `λ=4p/3`; `dephase(p)` → `λ=2p`; `amplitude_damping(γ)` maps 1:1. `pauli_channel`/`depolarize2` have no jl-native equivalent, so `runner.jl` builds each as one diagonal-PTM `TransferMapGate` (composing separate single-Pauli gates would add truncation points, breaking P5). Verified term-by-term in `test_gate_vocabulary_parity`; the runner range-checks the mapped `λ`. |
| P7 | `TransferMapGate` matrix ordering (`unitary_1q`/`unitary_2q`) | `qinds[1]` is the first (most significant) tensor factor of the matrix — pinned against `CliffordGate(:CNOT,[1,2])` — so the schema's "`matrix` acts on `\|q0 q1⟩`" maps to `qinds=[q0+1, q1+1]` verbatim. `calculateptm` defaults to `heisenberg=true` (the PTM of `mat'`). Confirmed term-by-term in both `[q0,q1]` and `[q1,q0]` orderings. |
| P8 | diagonal-PTM basis order | A `4^n × 4^n` matrix is taken as a PTM verbatim (no Heisenberg conjugation needed — diagonal, self-adjoint). 1-qubit order is `(I,X,Y,Z)`; 2-qubit index is `code(qinds[1]) + 4·code(qinds[2])`. What `pauli_channel` and `depolarize2` are built on. |
| P9 | exact-zero coefficients — **disagree** | With `min_abs_coeff=0.0`, jl keeps an exactly-zero coefficient; this engine's merge kernels and `from_strings` drop exact zeros unconditionally, so a circuit whose merge cancels exactly diverges by term count. `test_known_divergence_exact_zero` pins it with `amplitude_damping(γ=1)` (`X → √(1−γ)·X = 0` bit-exact): this engine keeps 0 terms, jl keeps 1. Mitigation: use a strictly positive `min_abs_coeff` and say so in the results file. |

## Known gaps

Named dependencies, not silent approximations.

* `direction="forward"` with `unitary_1q`, `unitary_2q`, `amplitude_damping`, `pauli_channel`,
  `depolarize2` — PauliPropagation.jl 0.8.2 defines no `_toschrodinger` for `TransferMapGate` or
  `AmplitudeDampingNoise`, so the runner rejects such a task up front
  (`test_forward_direction_rejects_unsupported_gates`) rather than dying inside `propagate`. Every
  Part A benchmark is Heisenberg, so nothing on the current branch needs this.
* `run.state = "y+"`, and per-qubit labels containing `+ - r l` — jl's `stateoverlap.jl` provides only
  `|0…0⟩`, `|+…+⟩`, `|1…1⟩` and computational basis states, not `|±i⟩`. This repo's A4 label alphabet
  is strictly larger, so a non-uniform non-computational product state cannot be compared against jl.
* `circuit.stim_file` — hard error, as above.
* `PP_FUSED=1` — no parity established (it truncates during gate application).
* `topn` truncation is absent from schema v1 on purpose: jl has no equivalent, and it is banned from
  comparative runs.
* `max_freq` / `max_sins` — jl-only truncations, excluded from the schema for the same reason.

## Coefficient type

The runner builds a `Float64` `PauliSum` when every observable coefficient is real, `ComplexF64`
otherwise, and reports which in `config.coeff_type`. The Hermitian-Y convention keeps real
coefficients real under every gate in the vocabulary, so timing numbers should be taken with a real
observable, and a complex one noted as such.

## Result JSON

One line on stdout (or `-o path`); diagnostics go to stderr, so stdout is always parseable.

```jsonc
{
  "runner": "benchmarks/julia/runner.jl", "schema_version": 1, "task_file": "...",
  "versions": {"julia": "1.12.6", "PauliPropagation": "0.8.2", "JSON3": "1.14.3"},
  "task":   {"n_qubits":…, "n_gates":…, "direction":…, "truncation":{…},
             "requested_threads":…, "state":…},
  "config": {"backend":"dict", "fused":false, "warm_repeats":3, "julia_threads":1,
             "coeff_type":"Float64", "min_abs_coeff_passed":…, "max_weight_passed":…},
  "result": {"expectation": {"re":…, "im":…} | null, "expectation_method": "overlapwithzero",
             "input_terms":…, "final_terms":…, "per_layer_terms":[…] | null,
             "peak_terms":… | null, "terms": {…} | null},
  "timing": {"wall_cold_s":…, "wall_warm_s":…, "wall_warm_all_s":[…],
             "gc_warm_s":…, "bytes_warm":…},
  "memory": {"vmrss_start_kb":…, "vmrss_pre_propagate_kb":…,
             "vmrss_post_propagate_kb":…, "vmhwm_kb":…, "source":…},
  "host":   {"hostname":…, "cpu":…, "ncores":…},
  "notes":  ["…"]
}
```

`wall_cold_s` is the first propagation and includes Julia's JIT; `wall_warm_s` is the minimum over
`PP_WARM_REPEATS` compiled runs. Per-layer counts come from a separate, untimed propagation, because
`@countpaulis` takes a lock per gate — never fold them into a timed run. `peak_terms` is
`max(input_terms, per_layer_terms…)`, matching the A2 peak-resident-count definition on the Rust side.

### Memory

The `memory` block is sampled from this process's own `/proc/self/status`, the only per-process
source — `getrusage(RUSAGE_CHILDREN)` from a Python driver conflates every reaped child, so it must
never be used for cross-engine memory comparison. Both engines sample themselves, in their own
process.

| field | meaning |
|---|---|
| `vmrss_start_kb` | Julia runtime + loaded packages, before the task is parsed — this engine's fixed per-process floor (~0.6 GiB on ccqlin038) |
| `vmrss_pre_propagate_kb` | after the circuit and observable are built, before any propagation |
| `vmrss_post_propagate_kb` | resident set after the timed propagations |
| `vmhwm_kb` | peak resident set over the process lifetime, sampled before the `@countpaulis` pass so that extra untimed propagation doesn't inflate it |
| `source` | `/proc/self/status`, or `"unavailable"` off Linux |

Per-term figures should subtract `vmrss_start_kb` as the fixed floor and divide the remainder by the
term count; report both the raw peak and the floor-subtracted figure, since at small term counts the
floor dominates completely. `notes` restates the P3/P4/P5/P9 findings in every result file, so a
results archive carries its own semantics caveats.
