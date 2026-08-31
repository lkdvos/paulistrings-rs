# PauliPropagation.jl baseline

Out-of-CI, subprocess-driven baseline for cross-engine comparisons. Nothing here is imported by the
Python package or by CI; the only entry points are a Julia script and a `subprocess` wrapper.

| file | role |
|---|---|
| `Project.toml`, `Manifest.toml` | pinned environment: **PauliPropagation.jl 0.8.2**, JSON3 1.14.3, BenchmarkTools 1.8.0, resolved for **julia 1.12.6** |
| `runner.jl` | reads a task JSON (schema v1), propagates, emits a result JSON |
| `probes.jl` | the semantics probes whose output every claim below is taken from |
| `../python/julia_baseline.py` | `subprocess` wrapper: builds/validates the task JSON, invokes the runner, parses the result, skips cleanly with no `julia` |
| `../python/test_julia_parity.py` | **the parity gate** — blocks cross-engine timing (adapted plan global rule 2) |

```bash
julia --project=benchmarks/julia benchmarks/julia/runner.jl task.json          # -> stdout JSON
julia --project=benchmarks/julia benchmarks/julia/probes.jl                    # semantics probes
python benchmarks/python/julia_baseline.py --self-test                         # wrapper smoke
python benchmarks/python/test_julia_parity.py                                  # parity report
pytest benchmarks/python/test_julia_parity.py -q                               # parity gate
```

The first Julia run precompiles (~30 s at first use, then cached). `julia` is found on `PATH`, at
`$JULIA_BINARY`, or at `~/.juliaup/bin/julia`; Lmod also provides one (`module load julia`).

There is **no PyJulia / juliacall anywhere**, by decision: `bench_baseline.py` records the same
exclusion for `PauliStrings.jl`, and the adapted plan restates it as D6. Subprocess only.

## Interchange format

`runner.jl` and `paulistrings.interop.load_task` both implement schema v1
(`research/notes/2026-09-01-python-api-extensions.md` §A5) **verbatim** — same keys, same gate-name
vocabulary, Hermitian-Y convention. Unknown top-level keys, unknown gate names, unknown gate fields,
and missing required keys are hard errors on both sides; the schema is versioned instead of tolerant.

Two things the runner does *not* do, on purpose:

* `circuit.stim_file` is a hard error — PauliPropagation.jl has no Stim parser, so a Stim-sourced
  circuit must be expanded into an inline `gates` list on the Python side.
* `run.direction` is never defaulted. jl's own default is `heisenberg=true`, this repo's is
  `"forward"`; defaulting either way would silently pick a picture.

One gate object = one channel on both engines. That is not cosmetic — see §P5.

### Non-schema runner knobs

Environment variables, so the frozen schema stays frozen. They change how the run is executed, never
what it means.

| variable | default | effect |
|---|---|---|
| `PP_BACKEND` | `dict` | `dict` = jl's `PauliSum` (hash map); `vector` = `VectorPauliSum` (the array/threaded backend) |
| `PP_WARM_REPEATS` | `3` | timed warm propagations after the cold one; `wall_warm_s` is the minimum |
| `PP_LAYER_COUNTS` | `1` | collect per-gate term counts (`@countpaulis`) in an extra, **untimed** propagation |
| `PP_FUSED` | `0` | experimental fused rotation kernel (`src/Performance/fused_dict.jl`). It truncates *during* gate application, so term-count parity is **not** established for it |
| `PP_EMIT_TERMS` | `0` | also emit the evolved sum term-by-term when it has at most N terms (parity debugging) |

Julia's worker count is a command-line flag, not an environment variable, so `julia_baseline.py`
passes `-t{run.threads}`; the runner warns to stderr if `Threads.nthreads()` disagrees with the task.

## Parity gate result

`pytest benchmarks/python/test_julia_parity.py` — 32 tests, all passing on ccqlin038
(julia 1.12.6, PauliPropagation 0.8.2, `RAYON_NUM_THREADS=1`).

The headline case is a 6-qubit, 57-gate circuit (`h` ×6, then 3 × [5 × `cnot`, 6 × `rz`, 6 × `rx`]),
observable `1.0·Z₂ + 0.5·(Z₁Z₄)`, one gate per channel:

| direction | state | `min_abs_coeff` | `max_weight` | layers compared | final terms (rust / jl) | \|Δexpectation\| |
|---|---|---|---|---|---|---|
| heisenberg | z+ | 1e-4 | — | 57 | 3813 / 3813 | 5.55e-17 |
| heisenberg | x+ | 1e-4 | — | 57 | 3813 / 3813 | 0 |
| heisenberg | z+ | 1e-6 | 4 | 57 | 236 / 236 | 0 |
| heisenberg | z+ | — | — | 57 | 3881 / 3881 | 5.55e-17 |
| forward | z+ | 1e-4 | — | 57 | 3926 / 3926 | 0 |

**All 57 per-layer term counts are identical in every row** (not just the final count), and
expectations agree to ≤ 5.6e-17 against a 1e-12 bar. Term counts are compared in *application*
order on both sides: jl's `@countpaulis` records after every gate, and this engine's DEBUG line is
`layer {k}/{n}` with `k` counting application steps, so for `direction="heisenberg"` both lists run
backwards through the task file and line up index by index.

The truncated rows are checked to be non-vacuous: the same circuit with no policy keeps 3881 terms,
so the 1e-4 row really is exercising coefficient truncation and the `max_weight=4` row really is
exercising weight truncation.

Beyond the headline circuit, every schema-v1 gate name gets its own single-gate task compared
**term by term** (coefficients to 1e-12, not just the contracted expectation, which is blind to a Y
sign that cancels): `h s x y z cnot cz swap rz rx ry pauli_rotation depolarize dephase
amplitude_damping pauli_channel depolarize2 unitary_1q unitary_2q`, plus reversed-qubit variants of
`cnot` and `unitary_2q` to catch a transposed index. All identical — **every** gate name in the
vocabulary, with no exceptions.

## RESOLVED: `amplitude_damping` was transposed relative to the unitary channels

**Fixed in the core; this section is kept as the record of a real bug this baseline caught.** Until
the fix, `AmplitudeDamping::apply` and `::apply_adjoint` in `channel/noise.rs` were swapped relative
to the convention every other channel follows, so `direction="heisenberg"` applied the Schrödinger
channel `Φ` instead of its dual `Φ†`.

What was measured at the time (γ = 0.3, single `amplitude_damping` gate, 8-term 3-qubit observable):

| | map applied to the qubit's Pauli |
|---|---|
| jl, `heisenberg=true` | `I → I`, `X,Y → √(1-γ)·same`, `Z → (1-γ)Z + γI` |
| this engine, `direction="heisenberg"` (**before the fix**) | `I → I + γZ`, `X,Y → √(1-γ)·same`, `Z → (1-γ)Z` |
| this engine, `direction="forward"` (**before the fix**) | identical to jl's `heisenberg=true` |

Why that was an inconsistency, not a choice:

* For unitary channels, `Channel::apply` in this core is the **Schrödinger** conjugation `U P U†` —
  `channel/clifford.rs` documents `S: X → Y`, i.e. `S X S†` (and `S† X S = -Y`, which is what
  `direction="heisenberg"` produces; probes.jl §P2 confirms jl agrees). So for unitaries
  `apply_adjoint` — what `direction="heisenberg"` calls — is the Heisenberg dual.
* `AmplitudeDamping` had it the other way round: `apply` was the **Heisenberg dual** `Φ†`
  (`I → I`, `Z → (1-γ)Z + γI`) and `apply_adjoint` was `Φ` itself (`I → I + γZ`).
* The Heisenberg dual of a trace-preserving channel is necessarily **unital** (`Φ†(I) = I`, because
  `Φ` preserves trace), so a Heisenberg map sending `I → I + γZ` cannot be a dual at all. Physically:
  `⟨Z⟩` for a qubit already in `|0⟩` — the fixed point of amplitude damping — decayed to `1-γ`
  instead of staying at `1`.
* `Depolarizing`, `Dephasing`, `PauliChannel` and `Depolarizing2Q` are self-adjoint, so the swap was
  invisible for them. `AmplitudeDamping` is the only built-in that exposes it.

The fix swapped the two bodies, so `apply` is now `Φ` (`I → I + γZ`, `Z → (1-γ)Z`) and
`apply_adjoint` is `Φ†` (`I → I`, `Z → (1-γ)Z + γI`), with the Kraus derivation written out in
`channel/noise.rs`. Measured after the fix, same fixture:

| direction | terms (rust / jl) | max coefficient \|Δ\| vs jl `heisenberg=true` |
|---|---|---|
| heisenberg | 9 / 9, labels identical | **0** (bit-exact on all 9) |
| forward | 11 / — | the transpose: no `III`, plus `ZXI ZZI ZIY` from the non-unital `I → I + γZ` |

`test_amplitude_damping_heisenberg_is_the_unital_dual` now pins the fixed orientation from both
sides, and `amplitude_damping` is back in `VOCAB_CASES` (the term-by-term sweep above).
`direction="forward"` still cannot be compared against jl — see "Known gaps".

## Semantics probes

Everything below is output of `julia --project=benchmarks/julia benchmarks/julia/probes.jl`
(PauliPropagation 0.8.2, julia 1.12.6). Each probe's expected value is hand-derived in a comment in
that file, never read back from the library. The paulistrings-side counterparts are the tests in
`../python/test_julia_parity.py`.

### P1. String and qubit-index convention

```
PauliString(3, :Z, 1) -> "ZII"
PauliString(3, :Z, 2) -> "IZI"
PauliString(3, :Z, 3) -> "IIZ"
getpauli codes: I=0 X=1 Y=2 Z=3
```

jl uses **1-based** qubit indices, and the leftmost character of a Pauli string is qubit 1. This
repo uses 0-based indices with the leftmost character qubit 0 (`from_strings({"ZII"})` sets the
z-bit of qubit 0). Same left-to-right order, so **observable keys map verbatim** and gate qubit
indices map with a `+1`. jl's internal 2-bit code (`I=0 X=1 Y=2 Z=3`) differs from this core's
`(x | z<<1)` packing (`I=0 X=1 Z=2 Y=3`); that is internal to each and never crosses the boundary.

### P2. Pauli-Y / coefficient convention — Hermitian on both sides

`S = diag(1, i)`, hand-derived with the Hermitian `Y = [[0,-i],[i,0]]`:
`S X S† = +Y` and `S† X S = -Y`.

```
S, heisenberg=true  : X -> [("Y", -1.0)]   coefftype=Float64
S, heisenberg=false : X -> [("Y",  1.0)]   coefftype=Float64
rz(0.3), heisenberg=true  : X -> [("X", 0.955336489125606), ("Y", -0.29552020666133955)]
rz(0.3), heisenberg=false : X -> [("X", 0.955336489125606), ("Y",  0.29552020666133955)]
```

PauliPropagation.jl is **Hermitian-Y**: a real coefficient multiplies the literal Pauli string, `Y`
carries no phase of its own, and `coefftype` stays `Float64` throughout. Identical to this repo's
convention (CLAUDE.md §Known gaps). The same numbers come out of this engine:
`s` with `direction="heisenberg"` gives `-1.0`, `rz(0.3)` gives `-sin(0.3)` on the `Y` term.

This also pins the **direction mapping**, which is exact for every gate in the vocabulary:

| task `run.direction` | jl kwarg | picture |
|---|---|---|
| `"heisenberg"` | `heisenberg=true` (jl's default) | reverse gate order, `U† O U` |
| `"forward"` | `heisenberg=false` | written gate order, `U O U†` |

jl "assumes gates are defined in the Heisenberg picture" and reverses the circuit; this engine's
`apply` is the Schrödinger conjugation and `direction="heisenberg"` reverses and calls
`apply_adjoint`. Different implementations, same observable map.

`test_hermitian_y_sign` encodes the sign as a cross-engine test: `[rz(θ), s]` in the Heisenberg
picture on `X`, contracted against `|+…+⟩`, is `-sin θ` on both engines. A phase-carrying `Y`, or an
`S` that mapped `X → +Y`, would flip that.

### P3. `min_abs_coeff` boundary — **the engines disagree exactly on the threshold**

`truncatemincoeff(coeff, min_abs_coeff) = abs(coeff) < min_abs_coeff` (`src/Base/truncate.jl`), so
jl **keeps** a coefficient exactly equal to the threshold. This repo's `CoefficientThreshold` keeps
`|c| > eps` (`truncation/builtin.rs:25`), so it **drops** it.

```
coeff=0.25              (== 0.25: true )  min_abs_coeff=0.25 -> 1 term(s) [("Z", 0.25)]
coeff=0.24999999999999994 (== 0.25: false)  min_abs_coeff=0.25 -> 0 term(s)
coeff=0.25000000000000006 (== 0.25: false)  min_abs_coeff=0.25 -> 1 term(s)
truncate!(psum with |c|=0.25; min_abs_coeff=0.25) -> 1 term(s)
```

The same three coefficients on this engine (`policy = coeff(0.25)`, `z` on a `Z` string, which is
the identity map with sign `+1` so nothing but truncation can move the coefficient):

```
coeff 0.25               == 0.25: True  -> 0 terms
coeff 0.24999999999999994 == 0.25: False -> 0 terms
coeff 0.25000000000000006 == 0.25: False -> 1 term
```

So the divergence is exactly one boundary case: **`|c| == eps` is kept by jl, dropped here.**

How much this matters: for generic angles it is a measure-zero event and every parity row above
passes untouched. It is *not* measure-zero for the two cases the plan actually calls for — dyadic
cutoffs (2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸ in benchmark C) with Clifford-point angles, where coefficients are exact
dyadics too and can land on the cutoff bit-exactly. Mitigation when it bites: perturb the threshold
on one side by one ulp and report that you did — never adjust a coefficient. `truncate` is applied
after every gate, so a boundary hit changes term counts for the whole rest of the run.

`test_known_divergence_coefficient_boundary` pins this so a version bump cannot change it silently.

### P4. `max_weight` boundary — the engines agree

`truncateweight(pstr, max_weight) = countweight(pstr) > max_weight`, and this repo's `WeightCutoff`
keeps `weight <= k`. Both keep `weight == max_weight`:

```
weights {1,2,3}, max_weight=2 -> [("ZII", 1.0), ("ZZI", 1.0)]
```

(this engine, same fixture: 2 terms kept). No mitigation needed.

### P5. When truncation is applied — **per gate, and there is no layer concept at all**

`_applymergetruncate!` (`src/Base/propagate.jl`) is `applytoall!` → `merge!` → `truncate!`, called
once per gate by `_propagate!`. There is no "layer" object in jl anywhere.

```
θ=0.05, min_abs_coeff=0.1, 2 gates -> per-gate counts = [1, 1]
same circuit, min_abs_coeff=0.0     -> per-gate counts = [1, 2]
```

`rz(0.05)` on `X` splits into `cos·X + sin·Y` with `sin = 0.04998 < 0.1 < cos`, so the `Y` branch
dies immediately and the *second* gate sees 1 term, not 2. Deferring truncation to the end of a
two-gate "layer" would have given `[2, 2]`.

This engine truncates after every channel, so the two coincide **iff one gate object is one
channel** — the adapted plan's D10 construction rule, which schema v1 makes structural (one gate
object, one `Circuit` push). Suite circuits must never fuse gates into a channel, and
`PauliNoise` gates in jl take a fused apply-and-truncate path
(`Propagation/specializations.jl:234`) that is still exactly one truncation point.

Count ordering, same probe:

```
counts for [rz, cnot] heisenberg=true  (applied cnot-then-rz) = [1, 2]
counts for [rz, cnot] heisenberg=false (applied rz-then-cnot) = [2, 2]
```

`@countpaulis` records in **application** order, so for `heisenberg=true` the list runs backwards
through the circuit as written. This engine's `layer {k}/{n}` DEBUG line numbers application steps
the same way, which is why the two lists compare index-by-index with no reversal.

### P6. Noise-channel parameter mapping

jl's `PauliNoise` channels damp by `1 - λ`; this repo's take a probability `p`. Derived and
confirmed:

```
X : Depolarizing(λ=4p/3=0.2) -> [("X", 0.8)]   | 1-4p/3 = 0.8
Y : Depolarizing(λ=4p/3=0.2) -> [("Y", 0.8)]
Z : Depolarizing(λ=4p/3=0.2) -> [("Z", 0.8)]
X : Dephasing(λ=2p=0.3)      -> [("X", 0.7)]   | 1-2p = 0.7 on X, Y
Y : Dephasing(λ=2p=0.3)      -> [("Y", 0.7)]
Z : Dephasing(λ=2p=0.3)      -> [("Z", 1.0)]   | Z untouched
```

| schema gate | jl gate | parameter map |
|---|---|---|
| `depolarize(p)` | `DepolarizingNoise(q, λ)` | `λ = 4p/3` (this repo's scale is `1 − 4p/3`) |
| `dephase(p)` | `DephasingNoise(q, λ)` (`= PauliZNoise`) | `λ = 2p` (this repo's scale is `1 − 2p`) |
| `amplitude_damping(γ)` | `AmplitudeDampingNoise(q, γ)` | 1:1 (the transpose bug above is fixed) |
| `pauli_channel(px,py,pz)` | one diagonal-PTM `TransferMapGate` | dual `I→1`, `X→1−2(py+pz)`, `Y→1−2(px+pz)`, `Z→1−2(px+py)` |
| `depolarize2(p)` | one diagonal-PTM `TransferMapGate` | dual `II→1`, all 15 others `→ 1−16p/15` |

jl has no native general Pauli channel or two-qubit depolarizing gate. Composing
`PauliXNoise ∘ PauliYNoise ∘ PauliZNoise` would be **three** gates and therefore three truncation
points, breaking the P5 parity rule; a single diagonal PTM is one gate with the exact dual, which is
why `runner.jl` builds them that way. Both are verified term-by-term against this engine's A6
channels in `test_gate_vocabulary_parity`. The runner range-checks the mapped `λ` and reports the
mapping in its output, so an out-of-range `p` fails loudly rather than producing a scale > 1.

### P7. `TransferMapGate` matrix ordering (`unitary_1q` / `unitary_2q`)

`TransferMapGate(mat, qinds)` takes a `2^n × 2^n` unitary in the 0/1 basis and calls
`calculateptm(mat)`, which defaults to `heisenberg=true` — i.e. the PTM of `mat'`, matching jl's
"gates are defined in the Heisenberg picture" convention. The two-qubit Kronecker order is
undocumented, so it was pinned against `CliffordGate(:CNOT, [1, 2])`:

```
2q X on q1: Clifford=[("XX", 1.0)]  ctrl-first-matrix=[("XX", 1.0)]  ctrl-second-matrix=[("XI", 1.0)]
2q X on q2: Clifford=[("IX", 1.0)]  ctrl-first-matrix=[("IX", 1.0)]  ctrl-second-matrix=[("XX", 1.0)]
2q Z on q1: Clifford=[("ZI", 1.0)]  ctrl-first-matrix=[("ZI", 1.0)]  ctrl-second-matrix=[("ZZ", 1.0)]
2q Z on q2: Clifford=[("ZZ", 1.0)]  ctrl-first-matrix=[("ZZ", 1.0)]  ctrl-second-matrix=[("IZ", 1.0)]
```

`qinds[1]` is the **first (most significant) tensor factor** of the matrix, so the schema's
"`matrix` acts on `|q0 q1⟩`" maps to `qinds = [q0+1, q1+1]` verbatim. Confirmed against this
engine's `unitary_2q` term-by-term, in both `[q0, q1]` and `[q1, q0]` orderings.

### P8. Diagonal-PTM basis order

A `4^n × 4^n` matrix is taken as a PTM verbatim (no Heisenberg conjugation — irrelevant for the
diagonal, self-adjoint case). The basis index is `symboltoint`'s:

```
diag PTM (1, .2, .3, .4) on I -> [("I", 1.0)]   X -> 0.2   Y -> 0.3   Z -> 0.4
2q diag PTM (idx1=0.5, idx4=0.25) on X at q1 -> [("XI", 0.5)]
2q diag PTM (idx1=0.5, idx4=0.25) on X at q2 -> [("IX", 0.25)]
```

1-qubit order is `(I, X, Y, Z)`; the 2-qubit index is `code(qinds[1]) + 4·code(qinds[2])`. This is
what `pauli_channel` and `depolarize2` are built on.

### P9. Exact-zero coefficients — a second, narrower divergence

With `min_abs_coeff = 0.0`, `abs(c) < 0` is never true, so jl **keeps** an exactly-zero coefficient:

```
single term with coeff 0.0, min_abs_coeff=0.0 -> 1 term(s)
```

This engine's merge kernels drop exact zeros unconditionally, and `from_strings` already drops a
zero coefficient at build time. So a circuit whose merge cancels *exactly* diverges by term count.
`test_known_divergence_exact_zero` pins it with `amplitude_damping(γ=1)`, whose `X → √(1−γ)·X = 0`
is bit-exact: this engine keeps 0 terms, jl keeps 1.

Not measure-zero in practice — Clifford-point angles produce exact cancellations. Mitigation for
comparative runs: use a strictly positive `min_abs_coeff` (any `eps > 0` kills jl's zeros too), and
say so in the results file. All parity rows above use either a positive threshold or a circuit with
no exact cancellation.

## Known gaps

Named dependencies, not silent approximations (adapted plan global rule 5).

* **`direction="forward"` with `unitary_1q`, `unitary_2q`, `amplitude_damping`, `pauli_channel`,
  `depolarize2`.** PauliPropagation.jl 0.8.2 defines no `_toschrodinger` method for
  `TransferMapGate` or `AmplitudeDampingNoise`, so it has no Schrödinger picture for them. The
  runner rejects such a task **up front**, naming the gap, rather than dying inside `propagate`
  (`test_forward_direction_rejects_unsupported_gates`). Every Part A benchmark is Heisenberg, so
  nothing on the current branch needs this. A fix would mean supplying the transposed transfer map
  ourselves — cheap for the diagonal (self-adjoint) noise PTMs, a small amount of work for a general
  unitary.
* **`run.state = "y+"`, and per-qubit labels containing `+ - r l`.** jl's `stateoverlap.jl` provides
  `|0…0⟩` (`overlapwithzero`), `|+…+⟩` (`overlapwithplus`), `|1…1⟩` and computational basis states
  (`overlapwithcomputational`); it says outright that "eval against `|±i⟩` [is] not implemented".
  So `"z+"`, `"x+"` and all-`0`/`1` label strings work, and everything else is a hard error. This
  repo's A4 label alphabet is strictly larger, so a non-uniform *non-computational* product state
  cannot be compared against jl at all.
* **`circuit.stim_file`** — hard error, as above.
* **`PP_FUSED=1`** — no parity established (it truncates during gate application).
* **`topn` truncation** is absent from schema v1 on purpose: jl has no equivalent, and it is banned
  from comparative runs (adapted plan D3).
* **`max_freq` / `max_sins`** — jl-only truncations, excluded from the schema for the same reason.

## Coefficient type

The runner builds a `Float64` `PauliSum` when every observable coefficient is real, and `ComplexF64`
otherwise, and reports which in `config.coeff_type`. `Float64` is what a jl user writes and is
roughly half the memory, and the Hermitian-Y convention keeps real coefficients real under every
gate in the vocabulary — so timing numbers should be taken with a real observable, and a complex one
noted as such.

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
  "host":   {"hostname":…, "cpu":…, "ncores":…},
  "notes":  ["…"]
}
```

`wall_cold_s` is the first propagation and includes Julia's JIT; `wall_warm_s` is the minimum over
`PP_WARM_REPEATS` compiled runs. Per-layer counts come from a **separate, untimed** propagation,
because `@countpaulis` takes a lock per gate — never fold them into a timed run. `peak_terms` is
`max(input_terms, per_layer_terms…)`, i.e. the peak *resident* count between gates, matching the A2
definition on the Rust side (the transient in-layer expansion is not observable on either engine).

`notes` restates the P3/P4/P5/P9 findings in every result file, so a results archive carries its own
semantics caveats.
