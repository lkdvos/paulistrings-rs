# D — XXZ chain scaling

<p class="lead">The only benchmark in the suite with an <em>analytic</em> prediction to check
against, and the one that finds the cross-engine ranking changes sign as the tracked set grows.
Below roughly 10⁴ tracked terms <code>PauliPropagation.jl</code> is 3–4× faster; above it this
engine is ~1.5× faster and pulling away. A single-point comparison would have "shown" either
engine winning by 3–4×.</p>

![Untruncated term count against Trotter steps, free and interacting regimes](../assets/xxz/term-growth.svg)

## Setup

The open XXZ chain

```text
H = Σ_{i=0}^{n-2} ( X_i X_{i+1} + Y_i Y_{i+1} + Jz Z_i Z_{i+1} )
```

is first-order-Trotterized at `dt = 0.1`: three `pauli_rotation` channels per bond, even bonds
then odd bonds, one gate per channel. Two regimes: `Jz = 0` (free) and `Jz = 0.5` (interacting).
Observables: the central `Z_c` (weight 1) and `Z_c Z_{c+1}` (weight 2). Direction is always
Heisenberg, initial state always a **domain wall** `|0…01…1⟩`.

`|0…0⟩` is an eigenstate of `H` at every `Jz`, so every expectation would be a constant; `|+…+⟩`
gives `⟨Z_c⟩ = 0` by symmetry. The domain wall is a computational basis state (so
`PauliPropagation.jl` can contract it too), and `⟨Z_c⟩` starts at exactly `−1` and moves. D's
deliverable is a scaling sweep (36 scaling points plus two truncation grids), which is why it
lives in `examples/xxz_chain/` rather than with the other benchmarks (no `pytest-benchmark` entry;
see Limitations).

## The `Jz = 0` growth law is quadratic

The measured log-log slope of untruncated non-zero term count against Trotter steps is
**exactly 2.0000** at `n = 40, 60, 80, 100`, weight-1 seed. The counts are `16 s²` exactly for
every unsaturated point, independent of `dt` (0.05, 0.1, 0.37 all identical) and of the seed site:

| steps `s` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| terms, `n = 100` | 16 | 64 | 144 | 256 | 400 | 576 | 784 | 1024 | 1296 | 1600 | 1936 | 2304 |
| `16 s²` | 16 | 64 | 144 | 256 | 400 | 576 | 784 | 1024 | 1296 | 1600 | 1936 | 2304 |
| terms, `n = 40` | 16 | 64 | 144 | 256 | 400 | 576 | 784 | 1024 | 1296 | 1600 | 1600 | 1600 |

The `n = 40` row is deliberate: past `s = 10` its light cone reaches both chain ends and the count
saturates (the law's boundary of validity), with a fit window of `s ≤ n/4 − 1`; a test asserts the
excluded tail breaks the slope. At `Jz = 0` the XX+YY chain is free-fermion: Jordan–Wigner maps it
to hopping Majoranas, and every Trotter gate is Gaussian, its conjugation acting on the Majorana
operators linearly with `O` orthogonal. A single-site `Z_c = −i γ_{2c} γ_{2c+1}` is a Majorana
bilinear, so it stays a sum of bilinears, and each `γ_a γ_b` is exactly one Pauli string. The
reachable `(a, b)` pairs number the square of the reachable Majorana indices, hence `O(s²)`, and
with this Trotter decomposition's cone exactly `(4s)² = 16 s²`. The CI test pins the slope, not
the counts: the prefactor 16 is a property of the bond-sweep ordering, not of the engine.

`Z_c Z_{c+1}` is a Majorana quartic, so the same argument gives `O(s⁴)`; measured counts are
exactly `(8s² + 6s − 1)²` (169, 1849, 7921, 22801, 52441, 104329, 187489, 312481), whose fitted
log-log slope over `2 ≤ s ≤ 8` is 3.70, the asymptotic 4 approached from below.
`Jz = 0.5` for contrast (same seed, `n = 40`, untruncated): 40 → 9512 → 2 453 872 terms at
`s = 1, 2, 3`, then exponential growth as the interaction breaks the bilinear closure — the regime
truncation exists for, and why D's large-`n` results are truncated and convergence-checked rather
than exact.

## Oracle: statevector at `n ≤ 26`

qiskit Aer dense statevector against the engine at `min_abs_coeff = 1e-12`, for both regimes, both
observables, `n ∈ {20, 24, 26}`, up to three Trotter steps — 30 cases. **All 30 cases agree**,
worst `|Δ|` = 2.2·10⁻¹² against a 10⁻⁹ bar (median 2.0·10⁻¹⁵). The two
worst cases are the deepest interacting ones (~818 000 terms after the 10⁻¹² cutoff), so the
error is the cutoff's, not a convention's; the free-regime shallow cases are at `0` or 2·10⁻¹⁶.
The two omitted case families (`n = 26` with `s = 3`, and `Jz = 0.5` with the weight-2 seed at
`s = 3`) are cost, not doubt: both keep ~4·10⁷ untruncated terms.

| n | Jz | observable | steps | statevector | paulistrings | \|Δ\| | terms |
|---|---|---|---|---|---|---|---|
| 20 | 0 | `Z_c` | 3 | −0.417930872482 | −0.417930872482 | 5.6e−17 | 144 |
| 20 | 0.5 | `Z_c` | 3 | −0.432856307369 | −0.432856307369 | 2.2e−12 | 818198 |
| 24 | 0 | `Z_cZ_{c+1}` | 3 | +0.393927398178 | +0.393927398178 | 1.2e−12 | 7247 |
| 24 | 0.5 | `Z_cZ_{c+1}` | 2 | +0.705482003672 | +0.705482003672 | 8.9e−14 | 146747 |
| 26 | 0.5 | `Z_c` | 2 | −0.731233222230 | −0.731233222230 | 2.4e−15 | 9504 |

A measured gotcha, worth knowing before adding any oracle-using timed run: one
`statevector_expectation` call takes the process from 32 to 97 threads (Aer's own OpenMP pool),
and the harness's thread pin counts threads gained since import, so any `threads=1` run after an
Aer call fails the assertion — both oracle-using modes therefore run every propagation first.

## Time and peak memory against `n` {#time-and-peak-memory-against-n}

![Warm propagation time and peak memory growth against chain length](../assets/xxz/time-memory-vs-n.svg)

6 Trotter steps, matched `min_abs_coeff = 1e-6` in both regimes, `n = 20…100`. One Python
subprocess per point, because `VmHWM` is a process-lifetime high-water mark: a single process
would report every later point's memory as the largest earlier one's. Warm propagation time (s),
one discarded warmup per point:

| n | channels | `Jz=0`, `Z_c` | `Jz=0`, `Z_cZ_{c+1}` | `Jz=0.5`, `Z_c` | `Jz=0.5`, `Z_cZ_{c+1}` |
|---|---|---|---|---|---|
| 20 | 342 | 0.0013 | 0.029 | 0.397 | 2.020 |
| 40 | 702 | 0.0023 | 0.052 | 0.613 | 2.771 |
| 60 | 1062 | 0.0034 | 0.080 | 0.847 | 3.396 |
| 80 | 1422 | 0.0049 | 0.114 | 1.194 | 4.874 |
| 100 | 1782 | 0.0064 | 0.143 | 1.474 | 5.759 |

Term count does not grow with `n` at all: ~257 (`Jz=0`, `Z_c`), ~7841 (`Jz=0`, weight 2),
~206 000 (`Jz=0.5`, `Z_c`), 517 000–822 000 (`Jz=0.5`, weight 2), essentially flat from `n = 20` to
`n = 100`. At fixed depth the operator's light cone, not the chain, sets the size of the tracked
set: six Trotter steps reach at most ~25 sites, so every `n ≥ ~30` run is the same physics padded
with identity. Time is nevertheless linear in `n` (`Jz=0.5`, `Z_c`: 0.40 s → 1.47 s for a 5.2×
channel count): the circuit has `3(n−1)` channels per step and the engine pays a pass over the sum
for each one regardless of support, so cost is `channels × terms`. A support-aware channel skip
would flatten this curve; nothing in the current engine has one. The counts alternate along the
sweep (7841 vs 6631; 821 750 vs 517 398): the seed sits on bond `c = n//2`, an even bond when
`n ≡ 0 (mod 4)` and odd otherwise, so the even-then-odd sweep hits it in a different half-step
each time, changing the truncation schedule (a property of the bond ordering, not noise). Peak
memory tracks the term count and the width monomorphization, and nothing else: for the `Jz=0.5`,
`Z_c` series (~206 000 terms at every `n`), growth is 11.2 MiB for `n ≤ 60` and 17.5 MiB for
`n ≥ 70`, **55 B/term against 87 B/term** — the `W = 1 → W = 2` boundary at 64 qubits (64-bit vs
128-bit symplectic keys: 32 B/term of key becomes 48 B/term).

## Convergence panels {#convergence-panels}

![Absolute error against runtime at n = 24](../assets/xxz/error-vs-runtime.svg)

`n = 24`, 6 Trotter steps, `Z_c`, against the exact statevector value:

| `min_abs_coeff` | `Jz=0` value | \|err\| | terms | s | `Jz=0.5` value | \|err\| | terms | s |
|---|---|---|---|---|---|---|---|---|
| 1e−2 | +0.1096220619 | 2.0e−2 | 53 | 0.0008 | +0.0548567031 | 3.0e−2 | 156 | 0.0017 |
| 1e−3 | +0.1287940647 | 5.4e−4 | 97 | 0.0034 | +0.0469663613 | 2.2e−2 | 1625 | 0.0097 |
| 1e−4 | +0.1293382772 | 3.3e−6 | 150 | 0.0040 | +0.0267695476 | 2.0e−3 | 9918 | 0.0491 |
| 1e−5 | +0.1293259793 | 9.0e−6 | 205 | 0.0042 | +0.0250209395 | 2.4e−4 | 48599 | 0.1216 |
| 1e−6 | +0.1293349316 | 1.8e−8 | 257 | 0.0045 | +0.0247991871 | 1.9e−5 | 206035 | 0.4409 |
| 1e−7 | +0.1293349390 | 1.0e−8 | 311 | 0.0034 | +0.0247800907 | 2.9e−7 | 776432 | 1.5256 |
| 1e−8 | +0.1293349428 | 6.5e−9 | 365 | 0.0037 | +0.0247793658 | 4.3e−7 | 2661871 | 4.9469 |

Exact reference: `⟨Z_c⟩ = +0.129334949228` (`Jz=0`), `+0.024779796796` (`Jz=0.5`). The free
regime reaches 10⁻⁸ with 365 terms and 4 ms; the interacting regime needs 2.7·10⁶ terms and 5 s to
reach 3·10⁻⁷. The error is not monotone in the cutoff (`Jz=0`: 3.3e−6 at 1e−4, then 9.0e−6 at 1e−5; `Jz=0.5`:
2.9e−7 at 1e−7, then 4.3e−7 at 1e−8), since dropped terms carry signs and can cancel, so a tighter
cutoff can land slightly further from the exact value while the trend still converges. The last
decade buys nothing measurable: below ~3·10⁻⁷ the comparison stops resolving, since 2.7·10⁶
floating-point coefficients summed in an unspecified order, contracted against a dense reference
with an error budget of its own, is the floor of the comparison, not of the truncation.

![Self-convergence at n = 60 and n = 100](../assets/xxz/self-convergence.svg)

The self-converged panels at `n = 60` and `n = 100` produce bit-identical values at every cutoff
(`+0.0247793658` at 10⁻⁸ for both) and identical term counts, differing only in wall time (9.6 s
vs 16.5 s at 10⁻⁸, the channel-count effect above): a 60-site chain and a 100-site chain are the
same calculation at six Trotter steps, so the `n = 100` self-converged answer is corroborated by
an `n = 24` run with an exact reference, where the same sweep converges to 4·10⁻⁷ of the
statevector value.

## Cross-engine timing {#cross-engine-timing-and-the-crossover}

Blocking parity gate first, then warm times: per-layer term counts are compared index by index in
gate-application order, and the task JSON is built from the same recorded gate list the engine
runs, so neither side gets a transcription of the other's circuit. Cutoffs are non-dyadic and
strictly positive (`1e-5`, `1e-6`), which avoids both measured divergences; no eps perturbation
was needed. **Parity: 4/4 cases pass**, on 171–702 layers each — every per-layer term count
identical index by index, final counts identical, expectations agreeing to ≤ 1.7·10⁻¹⁶ against a
10⁻¹² bar. Only then were the times below recorded (PauliPropagation.jl 0.8.2, julia 1.12.6,
`dict` backend, `-t1`, warm minimum of 3 repeats; this engine warm after one discarded run).

| n | Jz | steps | `min_abs_coeff` | layers | terms (both) | \|ΔE\| | paulistrings | jl warm | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 0 | 6 | 1e−6 | 702 | 257 | 8.3e−17 | 0.0076 s | 0.0018 s | **0.24×** |
| 20 | 0.5 | 3 | 1e−5 | 171 | 3 272 | 1.7e−16 | 0.0141 s | 0.0044 s | **0.31×** |
| 40 | 0.5 | 4 | 1e−6 | 468 | 29 745 | 5.6e−17 | 0.0735 s | 0.1068 s | **1.45×** |
| 40 | 0.5 | 6 | 1e−6 | 702 | 206 035 | 6.9e−18 | 0.6158 s | 0.9794 s | **1.59×** |

*("ratio" = jl warm / paulistrings; > 1 means this engine is faster.)* **The ranking changes sign
with the size of the tracked set**, somewhere between 3·10³ and 3·10⁴ terms: below the crossover
`PauliPropagation.jl`'s hash-map backend is 3–4× faster; above it this engine is ~1.5× faster and
pulling away. These circuits have `3(n−1)` channels per Trotter step (702 channels for the last
row), and this engine pays a bucketed per-layer pass per channel: a fixed cost a 257-term sum
cannot amortize. Both directions are far outside the ±5–8% single-threaded noise floor, so no A/B
protocol is needed here: the first three rows were measured twice and reproduced their ratios as
0.24/0.28, 0.31/0.32 and 1.45/1.48, same ranking and order of magnitude, well under the sign
changes being reported.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
# or one mode at a time: growth | statevector | scaling | convergence | julia | figures
pytest python/paulistrings/tests/test_benchmark_d_xxz.py    # the CI gate: 11 tests, ~4 s
```

Results are committed as `results/*.json`, one file per mode, overwritten on rerun.

## Limitations {#limitations}

No TDVP / tensor-network baseline exists at large `n`; no such package is a dependency of this
repository, so the interacting large-`n` numbers are self-converged and labelled as a follow-up
rather than silently approximated. Nor is there a `pytest-benchmark` entry: its per-point
recalibration would rerun each point for no gain, and the memory curve needs one process per point.

**Numbers:** every value on this page traces to
[`examples/xxz_chain/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/xxz_chain/README.md)
and the raw sweeps in `examples/xxz_chain/results/`. Host: ccqlin038, Intel Xeon Gold 6244 @ 3.60 GHz,
`RAYON_NUM_THREADS=1`, `RUST_LOG` unset.
