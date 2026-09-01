# Against other tools

<p class="lead">Two different questions. Against
<code>PauliPropagation.jl</code> — the other mature Pauli-propagation engine —
the question is "do these two compute the same thing, and which is faster
where?", and it is answered term for term. Against state-vector and stabilizer
simulators the question is "which tool is this?", and the answer is that they do
not overlap.</p>

## vs `PauliPropagation.jl`

The comparison baseline is subprocess-driven and out of CI, pinned to
**PauliPropagation.jl 0.8.2** on **julia 1.12.6** in a committed
`Project.toml`/`Manifest.toml`. There is deliberately **no PyJulia or juliacall
anywhere**: the only entry points are a Julia script that reads a task JSON and
emits a result JSON, and a `subprocess` wrapper that skips cleanly when no
`julia` is on `PATH`.

### Parity discipline

The rule is stated once and enforced everywhere: **term-count parity blocks
timing.** No cross-engine wall time may be reported for a configuration whose
evolved Pauli sums diverge term-for-term at matched truncation, so every timed
comparison in the suite runs the parity check first, untimed.

Four things make that check strong rather than decorative:

1. **One description, two engines.** Both sides are driven from the same
   schema-v1 task JSON, built from the *same recorded gate list* the
   `paulistrings` side runs — so neither engine gets a transcription of the
   other's circuit. Unknown keys, unknown gate names, unknown gate fields and
   missing required keys are hard errors on both sides; the schema is versioned
   instead of tolerant.
2. **Per applied layer, not just the final count.** A divergence that cancels by
   the end is exactly the coefficient-boundary or truncation-schedule bug the
   check exists to catch. Both engines report counts in application order, so for
   Heisenberg runs both lists walk backwards through the task file and line up
   index by index with no reversal.
3. **Term by term, not just the contracted expectation.** Every single gate name
   in the schema vocabulary — `h s x y z cnot cz swap rz rx ry pauli_rotation
   depolarize dephase amplitude_damping pauli_channel depolarize2 unitary_1q
   unitary_2q`, plus reversed-qubit variants of `cnot` and `unitary_2q` to catch a
   transposed index — gets its own single-gate task compared coefficient by
   coefficient to `1e-12`. All identical, no exceptions. The contracted
   expectation alone is blind to a `Y` sign that cancels.
4. **Along a sweep, not at one point.** "Identical counts at one cutoff" is a much
   weaker statement than "identical counts along a sweep", so the parity legs use
   three cutoffs.

The results, all from committed benchmark READMEs:

| where | configuration | compared | result |
|---|---|---|---|
| [Benchmark B](benchmarks/b-theta-sweep.md#cross-engine-parity) | 127 q, 5 steps, 3 observables × 3 cutoffs | 12 195 per-layer counts | **9/9 pass**, every count identical; 8 of 9 expectations agree to the last bit |
| [Benchmark C](benchmarks/c-deep-trotter.md#cross-engine-parity-at-the-deepest-point) | 127 q, **20 steps**, 3 dyadic cutoffs | 16 260 per-layer counts | **3/3 pass**, final *and* peak counts exact, expectations ≤ 5.6e-17 |
| [Benchmark D](benchmarks/d-xxz-chain.md#cross-engine-timing-and-the-crossover) | XXZ chain, 4 configurations | 171–702 layers each | **4/4 pass**, expectations ≤ 1.7e-16 |
| [Benchmark E](benchmarks/e-su4-brickwork.md#cross-engine-comparison) | Haar SU(4) brickwork, `unitary_2q` gates | all layers at two sizes | **exact**, and the expectation to 1e-12 |
| parity gate itself | 6 q, 57 gates, both directions, with and without truncation | 57 layers per row, 5 rows | **all identical**, expectations ≤ 5.6e-17 |

That last row is worth a note: the truncated rows are checked to be
**non-vacuous** — the same circuit with no policy keeps 3881 terms, so the
`1e-4` row really is exercising coefficient truncation and the `max_weight=4`
row really is exercising weight truncation.

### The conventions agree

Both engines are **Hermitian-Y**: a real coefficient multiplies the literal Pauli
string, `Y` carries no phase of its own, and the coefficient type stays real
under every gate in the vocabulary. That was verified by hand-derived probe, not
assumed — `S X S† = +Y` and `S† X S = −Y` come out identically on both sides, and
a cross-engine test encodes the sign so that a phase-carrying `Y`, or an `S` that
mapped `X → +Y`, would flip it.

Index conventions differ and map cleanly: jl uses 1-based qubit indices with the
leftmost character of a Pauli string being qubit 1, this repository uses 0-based
with the leftmost character qubit 0. Same left-to-right order, so **observable
keys map verbatim** and gate qubit indices map with a `+1`. The internal 2-bit
Pauli codes differ too, and never cross the boundary.

Direction maps exactly: `"heisenberg"` ↔ `heisenberg=true` (jl's default),
`"forward"` ↔ `heisenberg=false`. jl assumes gates are defined in the Heisenberg
picture and reverses the circuit; this engine's `apply` is the Schrödinger
conjugation and `"heisenberg"` reverses and calls `apply_adjoint`. Different
implementations, same observable map.

### The semantic divergences, measured

These were established by running probes whose expected values are hand-derived
in comments, never read back from the library.

#### The one real divergence: the coefficient boundary {#the-one-real-divergence}

jl truncates on `abs(coeff) < min_abs_coeff`, so it **keeps** a coefficient
exactly equal to the threshold. This repository's `CoefficientThreshold` keeps
`|c| > eps`, so it **drops** it. Measured on both sides with the same three
coefficients:

| coefficient | `== 0.25`? | jl at `min_abs_coeff = 0.25` | this engine |
|---|---|---|---|
| `0.25` | true | **1 term** | **0 terms** |
| `0.24999999999999994` | false | 0 terms | 0 terms |
| `0.25000000000000006` | false | 1 term | 1 term |

So the divergence is exactly one boundary case. For generic angles it is a
measure-zero event and every parity row above passes untouched. It is **not**
measure-zero for dyadic cutoffs at Clifford angles, where coefficients are exact
dyadics too and can land on the cutoff bit-exactly — which is why Benchmark B
could use powers of ten and ignore it and
[Benchmark C could not](benchmarks/c-deep-trotter.md#the-dyadic-cutoffs-and-the-one-ulp-mitigation).

**The mitigation, when it bites:** perturb the *threshold* on one side by one ulp
and report that you did — never adjust a coefficient. jl gets
`nextafter(eps, ∞)`, and since there is no float strictly between `eps` and that,
jl's `|c| < eps′` becomes exactly this engine's `|c| <= eps`, bit for bit, with no
coefficient touched. Truncation is applied after every gate, so a boundary hit
changes term counts for the whole rest of the run — this is not a cosmetic
concern. A test pins the divergence so a version bump cannot change it silently.

#### The second, narrower divergence: exact zeros

With `min_abs_coeff = 0.0`, `abs(c) < 0` is never true, so jl **keeps** an
exactly-zero coefficient. This engine's merge kernels drop exact zeros
unconditionally, and the builder drops a zero coefficient at build time. So a
circuit whose merge cancels *exactly* diverges by term count — pinned with
`amplitude_damping(γ=1)`, whose `X → √(1−γ)·X = 0` is bit-exact: this engine keeps
0 terms, jl keeps 1.

Not measure-zero in practice, since Clifford-point angles produce exact
cancellations. Mitigation for comparative runs: use a strictly positive
`min_abs_coeff` (any `eps > 0` kills jl's zeros too) and say so in the results
file.

#### Where the engines agree, and it was worth checking

- **The weight boundary.** jl truncates on `countweight > max_weight` and this
  repository's `WeightCutoff` keeps `weight <= k`; both keep
  `weight == max_weight`. No mitigation needed.
- **When truncation is applied.** jl calls apply → merge → truncate once **per
  gate**; there is no "layer" object in jl anywhere. This engine truncates after
  every channel, so the two coincide **iff one gate object is one channel** —
  which is why the suite's construction rule is one gate per `Circuit.push`, and
  why the schema makes it structural. Measured: `rz(0.05)` on `X` with
  `min_abs_coeff = 0.1` splits into `cos·X + sin·Y` with `sin < 0.1 < cos`, so the
  `Y` branch dies immediately and the *second* gate sees 1 term, not 2. Deferring
  truncation to the end of a two-gate "layer" would have given different counts.
- **Noise-channel parameter scales.** jl's Pauli-noise channels damp by `1 − λ`
  while this repository's take a probability `p`, so `depolarize(p)` maps to
  `λ = 4p/3` and `dephase(p)` to `λ = 2p`; `amplitude_damping(γ)` is 1:1. jl has
  no native general Pauli channel or two-qubit depolarizing gate, and composing
  three single-Pauli noise channels would be *three* gates and therefore three
  truncation points, breaking the parity rule — so the runner builds each as a
  single diagonal-PTM gate with the exact dual, one gate, one truncation point.
  Both verified term by term.
- **Two-qubit matrix ordering** for `unitary_2q` is undocumented upstream, so it
  was pinned against a known CNOT and confirmed in both qubit orderings.

### Known gaps — named, not approximated

- **`direction="forward"` with `unitary_1q`, `unitary_2q`, `amplitude_damping`,
  `pauli_channel`, `depolarize2`.** PauliPropagation.jl 0.8.2 defines no
  Schrödinger transfer map for those, so it has no forward picture for them. The
  runner rejects such a task **up front**, naming the gap, rather than dying
  inside `propagate`. Every benchmark in the suite is Heisenberg, so nothing
  currently needs it.
- **Non-computational, non-uniform product states.** jl provides `|0…0⟩`,
  `|+…+⟩`, `|1…1⟩` and computational basis states, and says outright that
  evaluation against `|±i⟩` is not implemented. This repository's per-qubit label
  alphabet is strictly larger, so such a state cannot be compared against jl at
  all.
- **Stim-sourced circuits** must be expanded into an inline gate list on the
  Python side — jl has no Stim parser, and the runner makes that a hard error
  rather than a silent path.
- **`topn` truncation** is absent from the interchange schema on purpose: jl has
  no equivalent, and it is banned from comparative runs. Likewise jl's own
  `max_freq` / `max_sins` truncations are excluded.
- **jl's experimental fused rotation kernel** has no parity established, because
  it truncates *during* gate application.

### A real bug the baseline caught {#a-real-bug-the-baseline-caught}

Worth recording, because it is the argument for keeping a cross-engine baseline at
all. Until it was fixed, `AmplitudeDamping::apply` and `::apply_adjoint` in the
core were **swapped** relative to the convention every other channel follows, so
`direction="heisenberg"` applied the Schrödinger channel `Φ` instead of its dual
`Φ†`.

That was an inconsistency, not a choice, and the argument is short: the Heisenberg
dual of a trace-preserving channel is necessarily **unital** (`Φ†(I) = I`), so a
Heisenberg map sending `I → I + γZ` cannot be a dual at all. Physically, `⟨Z⟩` for
a qubit already in `|0⟩` — the fixed point of amplitude damping — decayed to
`1−γ` instead of staying at `1`. The other four noise channels are self-adjoint,
so the swap was invisible for them; `amplitude_damping` is the only built-in that
exposes it.

After the fix, the same fixture gives 9 terms on both engines with identical
labels and **bit-exact** coefficients. A test now pins the orientation from both
sides, and [Showcase B2](showcases/b2-noisy-verification.md#the-same-collapse-three-other-channels)
carries the physics that fix turned on.

### Performance: it depends on the size of the tracked set

There is no single ratio, and reporting one would be misleading. Three committed
measurements, in increasing tracked-set size:

| source | configuration | tracked set | result |
|---|---|---|---|
| [Benchmark D](benchmarks/d-xxz-chain.md#cross-engine-timing-and-the-crossover) | XXZ, `n = 40`, `Jz = 0`, 702 layers | 257 terms | **jl 4.2× faster** (ratio 0.24×) |
| [Benchmark D](benchmarks/d-xxz-chain.md#cross-engine-timing-and-the-crossover) | XXZ, `n = 20`, `Jz = 0.5`, 171 layers | 3 272 terms | **jl 3.2× faster** (ratio 0.31×) |
| [Benchmark E](benchmarks/e-su4-brickwork.md#cross-engine-comparison) | Haar SU(4), `n = 10`, depth 5 | 381 654 terms | **within noise** (0.496 s vs 0.492 s) |
| [Benchmark D](benchmarks/d-xxz-chain.md#cross-engine-timing-and-the-crossover) | XXZ, `n = 40`, `Jz = 0.5`, 468 layers | 29 745 terms | **this engine 1.45× faster** |
| [Benchmark D](benchmarks/d-xxz-chain.md#cross-engine-timing-and-the-crossover) | XXZ, `n = 40`, `Jz = 0.5`, 702 layers | 206 035 terms | **this engine 1.59× faster** |
| [Benchmark C](benchmarks/c-deep-trotter.md#wall-time-reported-not-claimed) | 127 q kicked Ising, 20 steps, 5 420 layers | 8·10³ → 2.4·10⁶ terms | this engine 1.4×, 2.3×, 2.3× faster |

> **The ranking changes sign somewhere between 3·10³ and 3·10⁴ tracked terms.**
> Below the crossover jl's hash-map backend wins by 3–4×; above it this engine
> wins by ~1.5× and pulls away.

The mechanism is not mysterious. These circuits have `3(n−1)` channels per Trotter
step, and this engine pays a bucketed per-layer pass per channel — a fixed cost
that a 257-term sum cannot amortize, and one that a 2·10⁵-term sum amortizes
easily while the bucketed layout's cache behaviour and write-disjoint parallelism
start to matter. **A single-point comparison would have "shown" either engine
winning by 3–4×**, which is precisely why four sizes are reported.

Two caveats on the timing numbers, both from the source READMEs:

- The D ratios are far outside the ±5–8% single-thread noise floor, and the first
  three rows were measured twice, reproducing as 0.24/0.28, 0.31/0.32 and
  1.45/1.48 — same ranking, spread well under the sign changes being reported.
  Those are the solid ones.
- The C numbers are a **single warm repeat** per point on a shared workstation. A
  ~2.3× gap is well outside the noise band and the direction is consistent across
  three points spanning two orders of magnitude, so it is recorded — but the
  numbers to quote from C are its term counts and accuracy rows.

And one memory figure that **does not reproduce**: Benchmark B recorded jl's dict
backend at 67.6 GiB on a 2.85·10⁶-term sum, but Benchmark C re-measured the same
quantity with a per-process sampler and got 0.44–0.74 KiB/term plus ~0.7 GiB
fixed — 30–50× lower. The likely cause is named
([`getrusage(RUSAGE_CHILDREN)` conflating sibling children](benchmarks/c-deep-trotter.md#memory)),
and the recommendation is to re-measure before quoting either.

## vs state-vector simulation

Not a competitor — a **complementary oracle**, and this suite uses it as one
everywhere it reaches. Every exact reference on this site is a dense statevector
(usually qiskit Aer), and where two exact routes were affordable the reference is
*both* of them, required to agree.

The division of labour is a hard one:

| | state-vector | Pauli propagation |
|---|---|---|
| object carried | `2ⁿ` amplitudes | the number of Pauli strings the *observable* spreads over |
| cost driver | qubit count, full stop | circuit depth and how fast the operator spreads; `n` enters only through the channel count |
| result | any observable, exactly | one observable, to a truncation error you must measure |
| ceiling here | ~26–30 qubits (the 30-qubit cone reference cost ~150 s and 16.1 GiB) | 127 qubits routinely; 2.3·10⁸ terms in a single sum measured |

Two measured illustrations of why the boundary sits where it does:

- **The cone that broke the dense method.** Benchmark B needed an exact reference
  for a weight-10 observable whose causal cone is 30 qubits. Untruncated Pauli
  propagation over that cone was *aborted at a 26 GiB address-space cap* at 4.3·10⁸
  terms; the statevector over the same cone was ~150 s and does not care about
  depth. **The dense method won that one, decisively.**
- **The cone that broke the statevector.** The same benchmark's weight-17
  observable has a 59-qubit cone. `2^59` amplitudes rules out any dense method,
  and untruncated Pauli propagation is far past the wall above. **Neither method
  reaches it** — which is why those four references are self-converged and
  reported as not converged.

Also: **a state-vector simulator gives you the answer, and Pauli propagation
gives you the answer plus a truncation error you have to bound yourself.** That
asymmetry is why every page on this site carries a convergence panel, and why
"not claimable" appears as often as it does.

## vs stabilizer (`stim`) simulation

Also not a competitor, and also used here as an oracle. At a Clifford point
`stim` gives the exact ±1 integer in under 0.1 s, at any qubit count, and
[Benchmark A](benchmarks/a-clifford.md) exists to be scored against it. Benchmark
B reproduces those integers **bit-exactly at every one of eight cutoffs**, for
three observables at both Clifford endpoints.

What each tool is for:

- **`stim`** — Clifford circuits, exactly, at enormous scale. Nothing here
  competes with it in that regime, and where a circuit *is* Clifford it is
  strictly the better tool.
- **Pauli propagation** — non-Clifford circuits, where the tableau method has
  nothing to say. The kicked-Ising kick angle is exactly the knob that separates
  the two: at `θ_h ∈ {0, π/2}` the circuit is Clifford and `stim` answers; at the
  hard interior angles it is not, and the operator spreads over millions of Pauli
  strings.

One limitation worth naming because the suite ran into it: **a stabilizer
simulator cannot serve as a noisy oracle.** A tableau simulation samples one
Pauli error rather than averaging over them, so it is not a reference for a noise
channel — which is why [Showcase B2](showcases/b2-noisy-verification.md#validation-an-independent-dense-noisy-reference)
had to hand-roll a Kraus density-matrix reference instead of borrowing one.

## vs tensor-network / MPO methods

No measured head-to-head, and this site does not claim one. What it does have is
a *cost-model* comparison on the same operators:
[Showcase B6](showcases/b6-resource-probes.md) computes the Pauli-spectrum entropy
(the quantity governing truncation error for this engine) alongside the operator
entanglement across a bipartition (the quantity governing MPO bond dimension),
and finds them saying different things about the same operator — `S_2` grows
steadily with depth while `S_op` **saturates** around 1.3 nats from depth 5 on.
That is the Prosen–Pižorn observation in miniature, and it is the honest form of
the comparison available without a tensor-network dependency.

A TDVP baseline at large `n` is listed as
[a named follow-up in Benchmark D](benchmarks/d-xxz-chain.md#what-is-not-here),
not silently approximated.

## vs `qiskit.SparsePauliOp` / `openfermion.QubitOperator`

Different scope: these are Pauli-operator *containers* with algebraic
manipulation, not propagation engines with truncation. The repository benchmarks
construction and Clifford conjugation against both in
`benchmarks/python/bench_baseline.py` (parameterized over `n_terms ∈ {100, 1000,
10 000}` at 16 qubits), which is a manual, out-of-CI suite; those numbers are
regenerated by running it rather than committed.

`PauliStrings.jl` — the library that inspired this one — is deliberately excluded
from that comparison for the same reason `PauliPropagation.jl` is driven by
subprocess: no PyJulia wiring anywhere.

**Sources for this page:**
[`benchmarks/julia/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/julia/README.md)
(the probes, the divergences, the gaps, the bug record) and
[`benchmarks/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/README.md),
plus the per-benchmark READMEs linked inline above.
