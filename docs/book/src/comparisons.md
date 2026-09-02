# Against other tools

<p class="lead">Against <code>PauliPropagation.jl</code>, the other mature
Pauli-propagation engine, the comparison is term for term: same circuits, same
truncation, parity-gated timing. Against state-vector, stabilizer and MPO
methods the tools do not overlap, and this page records where each applies.</p>

## vs `PauliPropagation.jl`

The comparison baseline is subprocess-driven and out of CI, pinned to
**PauliPropagation.jl 0.8.2** on julia 1.12.6 in a committed
`Project.toml`/`Manifest.toml`. There is no PyJulia or juliacall anywhere: the
entry points are a Julia script that reads a task JSON and emits a result JSON,
and a `subprocess` wrapper that skips cleanly when no `julia` is on `PATH`.

### Parity discipline

Term-count parity blocks timing: no cross-engine wall time is reported for a
configuration whose evolved Pauli sums diverge term-for-term at matched
truncation, so every timed comparison runs the parity check first, untimed.

Four properties make the check strong:

1. One description, two engines. Both sides are driven from the same schema-v1
   task JSON, built from the same recorded gate list; unknown keys, gate names
   and fields are hard errors on both sides.
2. Per applied layer, not just the final count: a divergence that cancels by the
   end is exactly the truncation-schedule bug the check exists to catch.
3. Term by term, not just the contracted expectation. Every gate name in the
   schema vocabulary (plus reversed-qubit variants of `cnot` and `unitary_2q`)
   gets a single-gate task compared coefficient by coefficient to `1e-12`.
4. Along a sweep, not at one point: the parity legs use three cutoffs.

The results, all from committed benchmark READMEs:

| where | configuration | compared | result |
|---|---|---|---|
| [Benchmark B](benchmarks/b-theta-sweep.md#cross-engine-parity) | 127 q, 5 steps, 3 observables × 3 cutoffs | 12 195 per-layer counts | 9/9 pass, every count identical; 8 of 9 expectations agree to the last bit |
| [Benchmark C](benchmarks/c-deep-trotter.md#cross-engine-parity-at-the-deepest-point) | 127 q, 20 steps, 3 dyadic cutoffs | 16 260 per-layer counts | 3/3 pass, final and peak counts exact, expectations ≤ 5.6e-17 |
| [Benchmark D](benchmarks/d-xxz-chain.md#cross-engine-timing-and-the-crossover) | XXZ chain, 4 configurations | 171–702 layers each | 4/4 pass, expectations ≤ 1.7e-16 |
| [Benchmark E](benchmarks/e-su4-brickwork.md#cross-engine-comparison) | Haar SU(4) brickwork, `unitary_2q` gates | all layers at two sizes | exact, and the expectation to 1e-12 |
| parity gate itself | 6 q, 57 gates, both directions, with and without truncation | 57 layers per row, 5 rows | all identical, expectations ≤ 5.6e-17 |

The truncated rows of the last entry are non-vacuous: the same circuit with no
policy keeps 3 881 terms, so the `1e-4` row exercises coefficient truncation and
the `max_weight=4` row exercises weight truncation.

### The conventions agree

Both engines are Hermitian-Y: a real coefficient multiplies the literal Pauli
string, `Y` carries no phase of its own, and the coefficient type stays real
under every gate in the vocabulary. This was verified by hand-derived probe:
`S X S† = +Y` and `S† X S = −Y` come out identically on both sides, and a
cross-engine test encodes the sign.

Index conventions map cleanly: jl is 1-based with the leftmost Pauli character
qubit 1, this repository 0-based with the leftmost character qubit 0. Observable
keys map verbatim and gate indices map with a `+1`. Direction maps exactly:
`"heisenberg"` ↔ `heisenberg=true` (jl's default), `"forward"` ↔
`heisenberg=false`.

### The semantic divergences, measured

Established by probes whose expected values are hand-derived in comments, never
read back from the library.

#### The one real divergence: the coefficient boundary {#the-one-real-divergence}

jl truncates on `abs(coeff) < min_abs_coeff`, so it **keeps** a coefficient
exactly equal to the threshold; this repository's `CoefficientThreshold` keeps
`|c| > eps`, so it **drops** it. Measured on both sides:

| coefficient | `== 0.25`? | jl at `min_abs_coeff = 0.25` | this engine |
|---|---|---|---|
| `0.25` | true | 1 term | 0 terms |
| `0.24999999999999994` | false | 0 terms | 0 terms |
| `0.25000000000000006` | false | 1 term | 1 term |

For generic angles the divergence is a measure-zero event and every parity row
above passes untouched. It is not measure-zero for dyadic cutoffs at Clifford
angles, where coefficients are exact dyadics and can land on the cutoff
bit-exactly ([Benchmark C hits this](benchmarks/c-deep-trotter.md#the-dyadic-cutoffs-and-the-one-ulp-mitigation)).
The mitigation, when it bites: perturb the *threshold* on one side by one ulp
and report it, never a coefficient. jl gets `nextafter(eps, ∞)`; no float lies
between, so jl's `|c| < eps′` becomes exactly this engine's `|c| <= eps`. A test
pins the divergence so a version bump cannot change it silently.

#### The second, narrower divergence: exact zeros

With `min_abs_coeff = 0.0`, `abs(c) < 0` is never true, so jl keeps an
exactly-zero coefficient; this engine's merge kernels drop exact zeros
unconditionally. Pinned with `amplitude_damping(γ=1)`, whose
`X → √(1−γ)·X = 0` is bit-exact: this engine keeps 0 terms, jl keeps 1. Not
measure-zero in practice, since Clifford-point angles produce exact
cancellations. Mitigation for comparative runs: a strictly positive
`min_abs_coeff` (any `eps > 0` kills jl's zeros too), stated in the results
file.

#### Verified agreements

- The weight boundary: both engines keep `weight == max_weight`.
- When truncation is applied: jl truncates once per gate; this engine once per
  channel. The two coincide iff one gate object is one channel, which is the
  suite's construction rule and structural in the schema. Measured: `rz(0.05)`
  on `X` with `min_abs_coeff = 0.1` kills the `sin` branch immediately, so the
  second gate sees 1 term, not 2.
- Noise-channel parameter scales: jl damps by `1 − λ` where this repository
  takes a probability `p`, so `depolarize(p)` maps to `λ = 4p/3` and
  `dephase(p)` to `λ = 2p`; `amplitude_damping(γ)` is 1:1. jl has no native
  general Pauli channel or two-qubit depolarizing gate, so the runner builds
  each as a single diagonal-PTM gate with the exact dual — one gate, one
  truncation point. Both verified term by term.
- Two-qubit matrix ordering for `unitary_2q` is undocumented upstream; it was
  pinned against a known CNOT in both qubit orderings.

#### Noise-channel parity {#noise-channel-parity}

Noise-channel semantics agree between the engines term by term, including the
orientation of `amplitude_damping` under `direction="heisenberg"`: the
Heisenberg map is the unital dual `Φ†` (`Φ†(I) = I`), so `⟨Z⟩` for a qubit in
`|0⟩` — the channel's fixed point — stays at `1`. The shared fixture gives 9
terms on both engines with identical labels and bit-exact coefficients, a test
pins the orientation from both sides, and
[Showcase B2](showcases/b2-noisy-verification.md#the-same-collapse-three-other-channels)
carries the physics.

### Known gaps

- `direction="forward"` with `unitary_1q`, `unitary_2q`, `amplitude_damping`,
  `pauli_channel`, `depolarize2`: PauliPropagation.jl 0.8.2 defines no
  Schrödinger transfer map for those, and the runner rejects such a task up
  front. Every benchmark in the suite is Heisenberg.
- Non-computational, non-uniform product states: jl evaluates against
  `|0…0⟩`-style states only; this repository's per-qubit label alphabet is
  strictly larger, so such a state cannot be compared against jl.
- Stim-sourced circuits must be expanded into an inline gate list on the Python
  side; jl has no Stim parser, and the runner makes that a hard error.
- `topn` truncation is absent from the interchange schema: jl has no
  equivalent. Likewise jl's `max_freq` / `max_sins` truncations are excluded.
- jl's experimental fused rotation kernel has no parity established, because it
  truncates during gate application.

### Performance depends on the size of the tracked set

There is no single ratio: the ranking changes sign, and where it changes sign
depends on the workload by an order of magnitude.

The source is a dedicated head-to-head study
([`jl_performance/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/jl_performance/README.md)):
single-threaded core versus core, five interleaved `abba` pairs per
configuration, accepted on direction consistency, never on a difference of two
independently-noisy means. Every configuration passes a per-layer term-count
parity gate before any timing is reported. `ratio > 1` means this engine is
faster.

#### Where the ranking changes sign

| workload | channels | crossover (peak terms) |
|---|---|---|
| kicked-Ising, 127 q, 5 Trotter steps | 1 355 | **2.73 × 10³** (1.88 × 10³ with `engine="auto"`) |
| XXZ chain, n = 100, 6 Trotter steps | 1 782 | **2.00 × 10⁴** |
| Haar SU(4) brickwork, n = 36, depth 6 | 105 | none on the swept range: faster at every sign-consistent point |

The crossover spans 7× across these three workloads and the SU(4) matrix-gate
path has none at all, so no single global crossover is quoted anywhere on this
site.

#### Above the crossover

| workload | peak terms | ratio |
|---|---|---|
| kicked-Ising | 6.37 × 10⁵ | **2.146** |
| kicked-Ising | 2.15 × 10⁶ | 1.610 |
| XXZ | 2.66 × 10⁶ | **2.023**, still rising |
| Haar SU(4) | 2.30 × 10⁶ | **2.921**, still rising |

Memory, from the same study: process floors are 37.8 MB against Julia's
0.601 GiB, a factor of 16; at the largest SU(4) configuration peak RSS is
0.239 GiB against 1.625 GiB, or 95 floor-subtracted bytes per peak term against
479. Both engines sample their own `/proc/self/status`, never a driver-side
`getrusage(RUSAGE_CHILDREN)`.

#### Below the crossover

Below the crossover jl's hash-map backend is faster, by up to 3.6× at 68 terms:
a hash-map insert per term costs little at 10² terms, while the bucketed
per-layer pipeline costs nearly the same whatever the term count.

That fixed cost is avoidable. `propagate(engine="auto")` routes layers below
2 048 terms through a direct-apply path, worth 1.08–2.69× on exactly those
configurations
([`post-optimization-auto/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/jl_performance/post-optimization-auto/README.md)),
measured on the same binary against the default:

| workload | tracked set | `engine="sorted"` (default) | `engine="auto"` |
|---|---|---|---|
| XXZ | 1 625 terms | 0.372× (jl faster) | **1.040×, a measured tie** |
| XXZ | 9 918 terms | 0.873× (jl faster) | **1.051×, a measured tie** |
| Haar SU(4) | 1 416 terms | 1.097× | **1.660×** |
| kicked-Ising crossover | — | 2.73 × 10³ terms | **1.88 × 10³ terms** |

Above its threshold the path is inert, measured as its own control: SU(4) at
84 836 terms gives 1.409× with the path on and 1.416× with it off. All nine
configurations passed the per-layer parity gate with the path enabled — 9 618
per-layer counts, every one identical to PauliPropagation.jl's.

## vs state-vector simulation

Not a competitor but a complementary oracle, and this suite uses it as one
everywhere it reaches. Every exact reference on this site is a dense
statevector (usually qiskit Aer); where two exact routes were affordable the
reference is both of them, required to agree.

| | state-vector | Pauli propagation |
|---|---|---|
| object carried | `2ⁿ` amplitudes | the Pauli strings the *observable* spreads over |
| cost driver | qubit count | circuit depth and operator spreading; `n` enters only through the channel count |
| result | any observable, exactly | one observable, to a truncation error you must measure |
| ceiling here | ~26–30 qubits (the 30-qubit cone reference cost ~150 s and 16.1 GiB) | 127 qubits routinely; 2.3 × 10⁸ terms in a single sum measured |

Two measured illustrations of where the boundary sits:

- Benchmark B needed an exact reference for a weight-10 observable whose causal
  cone is 30 qubits. Untruncated Pauli propagation over that cone exceeded a
  26 GiB address-space cap at 4.3 × 10⁸ terms; the statevector over the same
  cone took ~150 s and does not care about depth.
- The same benchmark's weight-17 observable has a 59-qubit cone: `2^59`
  amplitudes rules out any dense method, and untruncated propagation is far
  past the wall above. Neither method reaches it, which is why those references
  are self-converged and reported as not converged.

A state-vector simulator gives the answer; Pauli propagation gives the answer
plus a truncation error to bound. That asymmetry is why every page on this site
carries a convergence panel.

## vs stabilizer (`stim`) simulation

Also an oracle, not a competitor. At a Clifford point `stim` gives the exact ±1
integer in under 0.1 s at any qubit count, and
[Benchmark A](benchmarks/a-clifford.md) exists to be scored against it.
Benchmark B reproduces those integers bit-exactly at every one of eight
cutoffs, for three observables at both Clifford endpoints.

- `stim`: Clifford circuits, exactly, at enormous scale. Where a circuit is
  Clifford it is strictly the better tool.
- Pauli propagation: non-Clifford circuits, where the tableau method has
  nothing to say. The kicked-Ising kick angle separates the two: at
  `θ_h ∈ {0, π/2}` the circuit is Clifford and `stim` answers; at the hard
  interior angles the operator spreads over millions of Pauli strings.

A stabilizer simulator cannot serve as a noisy oracle: a tableau simulation
samples one Pauli error rather than averaging over them, which is why
[Showcase B2](showcases/b2-noisy-verification.md#validation-an-independent-dense-noisy-reference)
carries a hand-rolled Kraus density-matrix reference instead.

## vs tensor-network / MPO methods

No measured head-to-head, and this site does not claim one. What it has is a
cost-model comparison on the same operators:
[Showcase B6](showcases/b6-resource-probes.md) computes the Pauli-spectrum
entropy (the quantity governing truncation error for this engine) alongside the
operator entanglement across a bipartition (the quantity governing MPO bond
dimension), and finds them saying different things about the same operator:
`S_2` grows steadily with depth while `S_op` saturates around 1.3 nats from
depth 5 on. A TDVP baseline at large `n` is
[a named limitation of Benchmark D](benchmarks/d-xxz-chain.md#limitations), not
silently approximated.

## vs `qiskit.SparsePauliOp` / `openfermion.QubitOperator`

Different scope: these are Pauli-operator containers with algebraic
manipulation, not propagation engines with truncation — no crossover concept
applies. The committed comparison
([`baseline_comparison/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/baseline_comparison/README.md))
benchmarks construction from string terms and one-layer Clifford conjugation,
seeded inputs, `n_terms ∈ {100, 1000, 10 000}`; medians in µs, ratio =
library / paulistrings:

| construct | paulistrings | qiskit | openfermion | qiskit ratio | openfermion ratio |
|---:|---:|---:|---:|---:|---:|
| 100 terms | 99.9 | 1 053.7 | 982.4 | 10.5× | 9.8× |
| 1 000 | 683.7 | 9 693.8 | 10 570.3 | 14.2× | 15.5× |
| 10 000 | 3 070.0 | 96 566.8 | 106 078.9 | 31.5× | 34.6× |

| conjugate by a Clifford layer | paulistrings | qiskit | ratio |
|---:|---:|---:|---:|
| 100 terms | 8.9 | 2 133.8 | 240× |
| 1 000 | 71.0 | 4 978.4 | 70× |
| 10 000 | 1 057.3 | 32 642.0 | 31× |

`openfermion` has no equivalent conjugation operation and is not in the second
group. `PauliStrings.jl`, the library that inspired this one, is excluded for
the same reason `PauliPropagation.jl` is driven by subprocess: no PyJulia
wiring anywhere.

**Sources for this page:**
[`benchmarks/julia/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/julia/README.md)
(the probes, the divergences, the gaps) and
[`benchmarks/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/README.md),
plus the per-benchmark READMEs linked inline. Cross-engine performance numbers:
[`jl_performance/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/jl_performance/README.md)
(protocol and headline tables, engine `81c568a`),
[`post-optimization/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/jl_performance/post-optimization/README.md)
and
[`post-optimization-auto/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/jl_performance/post-optimization-auto/README.md).
qiskit/openfermion numbers:
[`baseline_comparison/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/baseline_comparison/README.md)
and its committed `results.json`.
