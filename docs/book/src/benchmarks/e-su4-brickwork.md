# Benchmark E — Random SU(4) brickwork

<p class="lead">The deliberate worst case. Every other benchmark here is built
from a physically structured circuit with commuting layers, Clifford points, or
both — so truncation has real structure to exploit. This one draws an
<strong>independent Haar-random SU(4) block for every brickwork site</strong>:
no stabilizer structure, no commuting sublattice, no light-cone shortcut past
nearest-neighbour causality. Read every number below as "how does the engine
behave with no help from problem structure".</p>

![Peak and final term count against depth, at three truncation cutoffs](../assets/su4/term_count_vs_depth.svg)

## Setup

`n = 36` qubits, brickwork layers alternating on `(0,1),(2,3),…` and
`(1,2),(3,4),…`, an independent Haar-random SU(4) block per site, observable
`Z_18` (the central qubit), Heisenberg. Seed **20260831**, fixed everywhere in
the driver. Single-threaded.

## Oracle: statevector, where it reaches

`n ∈ {8, 12, 16, 20, 24}`, depth 3, **no truncation** — the operator's light cone
at depth 3 has at most 15 non-identity Pauli letters, so an untruncated
propagation is cheap at any of these sizes, and gives 4095 terms in every case
independently of `n`:

| n | oracle | engine | \|error\| |
|---|---|---|---|
| 8 | −0.0428035729 | −0.0428035729 | 1.25e-16 |
| 12 | +0.4658107269 | +0.4658107269 | 4.44e-16 |
| 16 | −0.1628731605 | −0.1628731605 | 6.66e-16 |
| 20 | −0.6075332763 | −0.6075332763 | 2.22e-16 |
| 24 | −0.1677957052 | −0.1677957052 | 2.16e-15 |

Agreement to double-precision tolerance at every size, against qiskit Aer's dense
statevector — two engines sharing no simulation code. The CI test re-derives this,
the truncated case, and every uniform product state as an `importorskip`ped gate.

## Term-count explosion, then collapse

There is **no Clifford savings anywhere in this circuit.** Two regimes are visible
in the numbers, not just the rising half:

- **Pre-saturation growth.** At the loosest truncation practical to run here
  (`min_abs_coeff = 1e-4`), peak term count goes 15 → 253 → 4038 → 58 079 →
  667 088 → 2 296 294 → 4 110 454 for depth 1…7 — roughly an order of magnitude
  every one to two layers. That is **faster than `4^depth`**, not slower: a
  generic 2-qubit Haar block's adjoint action is a dense ~15×15 real orthogonal
  map on the traceless Pauli algebra, not a 4-way branch.
- **Post-saturation collapse.** At looser cutoffs the *peak* plateaus quickly
  (~1300–1700 terms at `1e-2`, ~80 000–110 000 at `1e-3`) and then **the final
  term count falls to zero**: by depth 10 (`1e-2`) or depth 20 (`1e-3`), no
  coefficient anywhere in the operator's support survives the cutoff. This is not
  the propagation "giving up"; it is the expected fate of a local operator
  diffusing through a non-integrable, non-Clifford brickwork — eventually
  `⟨Z_18(t)⟩` really is, to truncation resolution, the maximally mixed value.

| min_abs_coeff | depth | final_terms | peak_terms | time (s) |
|---|---|---|---|---|
| 1e-2 | 1..6, 8 | 15 → 1483 (peak, depth 4) → 37 | 15..1483 | <0.02 each |
| 1e-2 | 10, 12, 16, 20 | **0** | 1258–1719 | <0.02 each |
| 1e-3 | 1..6 | 15 → 84 836 | 15..84 836 | 0.0006–0.65 |
| 1e-3 | 8, 10, 12, 16 | 32 567 → 1624 → 9 | 78 595–107 864 | 1.2–1.4 |
| 1e-3 | 20 | **0** | 85 174 | 1.4 |
| 1e-4 | 1..7 | 15 → 4 110 454 (still rising) | 15..4 110 454 | 0.0006–25.0 |

**Recorded cut.** The `1e-4` grid stops at depth 7 (25.0 s; depth 6 was already
7.9 s). Per-depth-step cost was measured growing ~2.3–3× at this cutoff, so depth
8 alone projects to roughly a minute and depth 9–10 to several minutes — out of
proportion for a benchmark classed `manual-short`. The two looser truncations
already show the full rise–plateau–collapse shape cheaply, so the qualitative
claim does not need the tighter grid extended.

## Error against runtime

![Absolute error against warm runtime at n = 16, depth 6](../assets/su4/error_vs_runtime.svg)

`n = 16`, depth 6 (statevector-checkable, and deep enough that fanout is well
underway). Oracle: −0.0497243601.

| min_abs_coeff | final_terms | time (s) | \|error\| |
|---|---|---|---|
| 1e-1 | 0 | 0.0002 | 4.97e-2 |
| 1e-2 | 391 | 0.008 | **8.11e-2** |
| 1e-3 | 62 569 | 0.44 | 1.75e-2 |
| 1e-4 | 2 257 674 | 6.82 | 1.75e-3 |
| 1e-5 | 12 539 219 | 14.5 | 1.57e-4 |
| 1e-6 | 16 251 799 | 15.3 | 5.41e-6 |
| 1e-8 | 16 771 992 | 17.9 | 7.02e-9 |

Error falls roughly with the cutoff itself — each decade in `eps` buys roughly a
decade of accuracy — until ~`1e-6`, where term count and runtime saturate: two
more decades of tightness add under 4% more terms and no measurable runtime, but
still buy three more decades of accuracy, because the residual error there is
dominated by *which specific* low-weight terms survive rather than by raw term
count.

Note the **non-monotone** row at `eps = 1e-2`: its error (8.1·10⁻²) is *larger*
than the empty-sum row above it (4.97·10⁻², which is just `|oracle|`). Truncation
error is not guaranteed monotone in the cutoff for a generic, non-sign-coherent
operator — which is exactly why this curve is reported as measured, not smoothed
or asserted monotone.

## Time and memory against `n`

![Time and peak memory against qubit count at fixed depth](../assets/su4/time_memory_vs_n.svg)

Depth 6, `min_abs_coeff = 1e-4`, `n` from 8 to the 36-qubit headline.
Oracle-checked (statevector) up to `n = 24`; `n ∈ {28, 32, 36}` is unchecked and
labelled `oracle_checked: false` in the results JSON — self-converged only, per
the rule that every truncated claim beyond oracle range says so plainly rather
than borrowing an oracle check it did not get.

Wall time is dominated by term count, not `n` directly: this circuit's causal
structure means the light cone from the central qubit, not `n` itself, sets how
many terms accumulate by a fixed depth.

**Every point's `peak_memory_kb_delta` reads exactly `0.0`** — *not* because these
runs use no memory, but because this driver runs everything in one process and
`VmHWM` is a process-lifetime high-water mark: the error-vs-runtime sweep's
16.2–16.8 M-term points, run earlier in the same process, already set a higher
peak than any size-scaling point reaches. A size/memory figure that needs a real
per-point number would need one process per point (which is exactly what
[Benchmark D](d-xxz-chain.md#time-and-peak-memory-against-n) does); this driver
does not, **and says so rather than reporting a misleadingly flat `0.0` curve as
if it meant something.**

## Determinism

Same seed twice → identical term counts and expectation to 1e-12. Asserted in the
driver's own smoke re-run and, authoritatively, in
`test_benchmark_e_su4.py::test_same_seed_twice_gives_identical_term_counts_and_expectation`
— which always runs in CI, with no optional dependency — plus a negative control
confirming two distinct seeds generically disagree.

## Cross-engine comparison

Schema v1's `unitary_2q` gate is exactly the SU(4) block this circuit is built
from, and `PauliPropagation.jl` 0.8.2 defines no Schrödinger transfer map for a
`TransferMapGate`, so **only `direction="heisenberg"` is comparable** — the same
restriction the rest of the suite documents for any `unitary_1q`/`unitary_2q`/
noise gate against this version.

`min_abs_coeff = 1e-4` is deliberately **non-dyadic**: the one known
coefficient-boundary divergence is triggered by *exact* dyadic coefficients
landing on the cutoff, which requires a Clifford point. Every coefficient in a
Haar-random SU(4) circuit is an irrational float, so the boundary case cannot
arise here regardless — the non-dyadic choice removes it as a possible objection
anyway.

**Per-layer term-count parity holds exactly**, checked before any timing was
recorded:

| n | depth | rust final_terms | jl final_terms | parity |
|---|---|---|---|---|
| 6 (smoke) | 3 | 4051 | 4051 | exact, all layers |
| 10 (timed) | 5 | 381 654 | 381 654 | exact, all layers |

Both per-layer counts and the final expectation (to `1e-12`) matched at both
sizes, reusing the same parity gate the rest of the suite's comparisons use — so
this reconfirms the documented vocabulary parity specifically for the
`unitary_2q` gates this circuit is built from, rather than re-deriving a separate
comparison mechanism.

**Timed comparison** (`n = 10`, depth 5, single-threaded both sides, warm /
JIT-excluded on the jl side):

| engine | version | time (s) | final_terms |
|---|---|---|---|
| paulistrings | (see `results.json` provenance) | 0.496 | 381 654 |
| PauliPropagation.jl | 0.8.2 | 0.492 | 381 654 |

The two engines are **within measurement noise of each other** on this point
(±5–8% single-threaded is expected between *any* two runs on this host, let alone
two different implementations) — a parity-gated same-ballpark result, not a speed
claim in either direction. Repeated end-to-end runs during development landed
both in the 0.47–0.53 s range each time, always within a few percent of each
other, so the "same ballpark" reading is not a one-off.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_e_su4.py
pytest python/paulistrings/tests/test_benchmark_e_su4.py    # the CI gate
```

Seed 20260831 throughout; `results.json` carries the full per-run provenance
block (commit, CPU, rustc/julia/PauliPropagation.jl versions, thread count) for
every record. `RunRecord.provenance.thread_count` records `None` rather than a
possibly-misleading count on the runs that follow a qiskit-aer statevector call in
the same process — Aer's own thread pool defeats the harness's process-wide
thread-delta heuristic once it has run; `RAYON_NUM_THREADS=1`, asserted at the top
of the driver before anything propagates, is what actually pins the engine.

**Source for every number on this page:**
[`benchmarks/python/su4_staircase/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/su4_staircase/README.md),
with the raw records in `results.json` next to it.
