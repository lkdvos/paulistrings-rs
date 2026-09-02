# E — Random SU(4) brickwork

<p class="lead">The deliberate worst case. Every other benchmark here is built
from a physically structured circuit with commuting layers, Clifford points, or
both, so truncation has real structure to exploit. This one draws an
independent Haar-random SU(4) block for every brickwork site: no stabilizer
structure, no commuting sublattice, no light-cone shortcut past
nearest-neighbour causality. Every number below shows how the engine behaves
with no help from problem structure.</p>

![Peak and final term count against depth, at three truncation cutoffs](../assets/su4/term_count_vs_depth.svg)

## Setup

`n = 36` qubits, brickwork layers alternating on `(0,1),(2,3),…` and
`(1,2),(3,4),…`, an independent Haar-random SU(4) block per site, observable
`Z_18` (the central qubit), Heisenberg direction. Seed **20260831**, fixed
throughout. Single-threaded.

## Results

There is no Clifford savings anywhere in this circuit. Two regimes are visible
in the numbers:

- **Pre-saturation growth.** At `min_abs_coeff = 1e-4`, peak term count goes
  15 → 253 → 4038 → 58 079 → 667 088 → 2 296 294 → 4 110 454 for depth 1…7 —
  roughly an order of magnitude every one to two layers, **faster than
  `4^depth`**: a generic 2-qubit Haar block's adjoint action is a dense
  ~15×15 real orthogonal map on the traceless Pauli algebra, not a 4-way
  branch.
- **Post-saturation collapse.** At looser cutoffs the peak plateaus quickly
  (~1300–1700 terms at `1e-2`, ~80 000–110 000 at `1e-3`) and the final term
  count then **falls to zero**: by depth 10 (`1e-2`) or depth 20 (`1e-3`), no
  coefficient in the operator's support survives the cutoff. This is the
  expected fate of a local operator diffusing through a non-integrable,
  non-Clifford brickwork — `⟨Z_18(t)⟩` reaches the maximally mixed value at
  truncation resolution.

| min_abs_coeff | depth | final_terms | peak_terms | time (s) |
|---|---|---|---|---|
| 1e-2 | 1..6, 8 | 15 → 1483 (peak, depth 4) → 37 | 15..1483 | <0.02 each |
| 1e-2 | 10, 12, 16, 20 | **0** | 1258–1719 | <0.02 each |
| 1e-3 | 1..6 | 15 → 84 836 | 15..84 836 | 0.0006–0.65 |
| 1e-3 | 8, 10, 12, 16 | 32 567 → 1624 → 9 | 78 595–107 864 | 1.2–1.4 |
| 1e-3 | 20 | **0** | 85 174 | 1.4 |
| 1e-4 | 1..7 | 15 → 4 110 454 (still rising) | 15..4 110 454 | 0.0006–25.0 |

The `1e-4` grid stops at depth 7 (25.0 s; depth 6 was 7.9 s) — per-step cost
grows ~2.3–3× at this cutoff, so depth 8 alone would cost roughly a minute.
The two looser truncations already show the full rise–plateau–collapse shape
cheaply.

### Error against runtime

![Absolute error against warm runtime at n = 16, depth 6](../assets/su4/error_vs_runtime.svg)

`n = 16`, depth 6 (statevector-checkable, deep enough for fanout to be well
underway). Oracle: −0.0497243601.

| min_abs_coeff | final_terms | time (s) | \|error\| |
|---|---|---|---|
| 1e-1 | 0 | 0.0002 | 4.97e-2 |
| 1e-2 | 391 | 0.008 | 8.11e-2 |
| 1e-3 | 62 569 | 0.44 | 1.75e-2 |
| 1e-4 | 2 257 674 | 6.82 | 1.75e-3 |
| 1e-5 | 12 539 219 | 14.5 | 1.57e-4 |
| 1e-6 | 16 251 799 | 15.3 | 5.41e-6 |
| 1e-8 | 16 771 992 | 17.9 | 7.02e-9 |

Error falls roughly a decade per decade of cutoff until `~1e-6`, where term
count and runtime saturate: two more decades of tightness add under 4% more
terms and no measurable runtime, but still buy three more decades of accuracy,
since the residual error there is dominated by *which* low-weight terms
survive rather than raw term count.

The `eps = 1e-2` row is **non-monotone**: its error (8.1e-2) is larger than the
empty-sum row above it (4.97e-2, just `|oracle|`). Truncation error is not
guaranteed monotone in the cutoff for a generic, non-sign-coherent operator.

### Time and memory against `n`

![Time and peak memory against qubit count at fixed depth](../assets/su4/time_memory_vs_n.svg)

Depth 6, `min_abs_coeff = 1e-4`, `n` from 8 to the 36-qubit headline.
Oracle-checked up to `n = 24`; `n ∈ {28, 32, 36}` is self-converged only
(`oracle_checked: false` in `results.json`). Wall time tracks term count, not
`n` directly: this circuit's causal structure means the light cone from the
central qubit sets how many terms accumulate by a fixed depth, not `n` itself.

Every point's `peak_memory_kb_delta` reads `0.0` in `results.json`. RSS is
cumulative across a process; the figure is an upper bound when other suites
ran first — a real per-point number needs one process per point, as in
[Benchmark D](d-xxz-chain.md#time-and-peak-memory-against-n).

## Cross-engine comparison

Schema v1's `unitary_2q` gate is exactly the SU(4) block this circuit is built
from, and `PauliPropagation.jl` 0.8.2 defines no Schrödinger transfer map for a
`TransferMapGate`, so only `direction="heisenberg"` is comparable — the same
restriction applies to any `unitary_1q`/`unitary_2q`/noise gate against this version.

`min_abs_coeff = 1e-4` is deliberately non-dyadic: the one known
coefficient-boundary divergence needs *exact* dyadic coefficients landing on
the cutoff, which requires a Clifford point. Every coefficient in a
Haar-random SU(4) circuit is an irrational float, so the boundary case cannot
arise here.

**Per-layer term-count parity holds exactly** — matched at both sizes,
including the final expectation to `1e-12`:

| n | depth | rust final_terms | jl final_terms | parity |
|---|---|---|---|---|
| 6 (smoke) | 3 | 4051 | 4051 | exact, all layers |
| 10 (timed) | 5 | 381 654 | 381 654 | exact, all layers |

**Timed comparison** (`n = 10`, depth 5, single-threaded both sides,
warm/JIT-excluded on the jl side):

| engine | version | time (s) | final_terms |
|---|---|---|---|
| paulistrings | (see `results.json` provenance) | 0.496 | 381 654 |
| PauliPropagation.jl | 0.8.2 | 0.492 | 381 654 |

The two engines are **within measurement noise of each other** (±5–8%
single-threaded is expected between any two runs on this host) — a
parity-gated same-ballpark result, not a speed claim in either direction.

## Validation

`n ∈ {8, 12, 16, 20, 24}`, depth 3, no truncation — the light cone at depth 3
has at most 15 non-identity Pauli letters, giving 4095 terms at every size.
Agreement is to double-precision tolerance against qiskit Aer's dense
statevector, two engines sharing no simulation code:

| n | oracle | engine | \|error\| |
|---|---|---|---|
| 8 | −0.0428035729 | −0.0428035729 | 1.25e-16 |
| 12 | +0.4658107269 | +0.4658107269 | 4.44e-16 |
| 16 | −0.1628731605 | −0.1628731605 | 6.66e-16 |
| 20 | −0.6075332763 | −0.6075332763 | 2.22e-16 |
| 24 | −0.1677957052 | −0.1677957052 | 2.16e-15 |

Same seed twice gives identical term counts and expectation to `1e-12`
(`test_benchmark_e_su4.py::test_same_seed_twice_gives_identical_term_counts_and_expectation`,
always in CI), with a negative control confirming distinct seeds generically
disagree.

**Numbers:** every figure on this page traces to
[`benchmarks/python/su4_staircase/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/su4_staircase/README.md),
including the run command, with the raw records in `results.json` next to it.
