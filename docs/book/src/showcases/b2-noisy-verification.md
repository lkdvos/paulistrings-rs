# B2 — Noisy circuit verification

<p class="lead">The 127-qubit heavy-hex kicked-Ising circuit from
<a href="../benchmarks/c-deep-trotter.html">Benchmark C</a> gets per-gate depolarizing
noise added, and the noise makes the simulation cheaper. At <code>p = 3e-2</code> the
tracked set peaks at 651× fewer terms than at <code>p = 0</code>, and the run finishes
1078× faster, at the same cutoff. Every density-matrix method scales the opposite
way.</p>

![Peak and final term count, and wall time, against the noise rate](../assets/b2/terms-and-time-vs-noise.svg)

## Noise shrinks the tracked set

A Kraus/density-matrix simulator pays for noise twice: it carries a `4ⁿ` object
instead of a `2ⁿ` one, and each channel costs more than a gate. Pauli propagation
pays negatively: a single-qubit depolarizing channel rescales a coefficient by
`1 − 4p/3` per non-identity qubit on its support, so a weight-`w` string crossing one
noise layer loses `(1 − 4p/3)^w`, and after `d` layers a string that held weight `w`
throughout is down by `(1 − 4p/3)^{wd}`. The filter is exponential in weight, so a
fixed `min_abs_coeff` becomes an effective weight cap that tightens with depth, and
the tracked set of a scrambling circuit is dominated by its high-weight tail. One
channel runs on every qubit in the support of the gate that just ran, 686 channels
per step at `n = 127`, 13 720 across 20 steps. `p = 0` runs as `depolarize(0.0)`
rather than a separate circuit, so the rescale is exactly `1.0` and the `p = 0` leg
is the same computation as the noiseless one, on the noisy legs' channel schedule —
pinned term for term by a test.

## Running it

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py           # ~26 min
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick   # ~1 s, 20 qubits, writes nothing
python examples/b2_noisy_verification/run_b2.py --figures-only                # re-render the SVGs
pytest python/paulistrings/tests/test_showcase_b2.py                          # the CI gate
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts; the driver
refuses to run otherwise. The driver adds one noise call per gate
(`circuit.depolarize`/`dephase`/`amplitude_damping`/`pauli_channel`) and propagates
with `harness.run_propagation(circuit, observable, TruncationSpec(min_abs_coeff=eps), direction="heisenberg")`.

## Terms and time across the noise sweep

`θ_h = 5π/16`, 20 Trotter steps, `Z_62`, Heisenberg, contracted against `|0…0⟩`,
single-threaded, fixed `min_abs_coeff = 2⁻¹⁴`:

| p | ⟨Z₆₂⟩ | final terms | peak terms | max weight | mean weight | wall (1 thread) | peak vs `p=0` |
|---|---|---|---|---|---|---|---|
| **0** | +0.016131374386 | 34 698 | **14 396 463** | 19 | 10.30 | **531.0 s** | — |
| 1e-3 | +0.012998050638 | 20 098 | 9 710 246 | 19 | 9.80 | 361.1 s | 1.5× fewer |
| 5e-3 | +0.006280876086 | 3 521 | 2 818 675 | 14 | 8.61 | 82.5 s | 5.1× fewer |
| 1e-2 | +0.002432260559 | 408 | 869 299 | 13 | 7.41 | 20.2 s | 16.6× fewer |
| **3e-2** | +0.000000000000 | **0** | **22 105** | 0 | 0 | **0.49 s** | **651× fewer** |

Maximum surviving weight falls 19, 19, 14, 13, 0 and mean weight 10.3, 9.8, 8.6, 7.4,
0, the exponential-in-weight filter acting on the tail directly. Wall time falls
faster than the term count (1078× against 651×) because the merge phase sorts a
resident sum far out of cache at the larger sizes, so a smaller tracked set pays
twice. A density-matrix method at `n = 127` is not on the table (`4^127`
amplitudes), so the comparison is within this engine: `p = 0` against the noisy
legs in the same sweep.

## Convergence across cutoffs

![Convergence against cutoff, one curve per noise rate](../assets/b2/convergence-vs-cutoff.svg)

At `p = 3e-2` the fixed-cutoff answer above reads exactly `0`; the converged answer is 7.8·10⁻⁵.

| p | 2⁻⁸ | 2⁻¹⁰ | 2⁻¹² | 2⁻¹⁴ | 2⁻¹⁶ | 2⁻¹⁸ | 2⁻²⁰ | last Δ | plateau? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.011941 | 0.014730 | 0.015481 | **0.016131** | — | — | — | 6.50e-4 | yes |
| 1e-3 | 0.007101 | 0.011707 | 0.012737 | **0.012998** | — | — | — | 2.61e-4 | no |
| 5e-3 | 0 | 0.005486 | 0.005918 | **0.006281** | — | — | — | 3.63e-4 | yes |
| 1e-2 | 0 | 0 | 0.002272 | 0.002432 | **0.002484** | — | — | 5.14e-5 | yes |
| 3e-2 | 0 | 0 | 0 | **0** | 7.156e-5 | 7.753e-5 | **7.795e-5** | 4.23e-7 | yes |

A fixed cutoff is not a fixed accuracy: as `p` grows the signal shrinks, so the
cutoff must tighten with it, and at `p = 3e-2` a `2⁻¹⁴` cutoff sits above every
contributing coefficient. Noise buys the cutoff reach that fixes this cheaply: the
whole `p = 3e-2` sweep to `2⁻²⁰` costs 76 s, an eighth of the single `p = 0` point at
`2⁻¹⁴`, resolving the answer to 4.2·10⁻⁷ against 6.5·10⁻⁴ at `p = 0`; at `p = 0` the
next dyadic pair is out of reach, projecting ~1.9·10⁸ terms and ~2 h single-threaded.
The plateau verdict reuses Benchmark B's criterion unchanged, and rejects `p = 1e-3`:
its last difference is 2.61·10⁻⁴ but its previous one is 1.03·10⁻³, just over
tolerance, so that leg is unresolved rather than quoted.

## The same collapse, three other channels

Same circuit, same `2⁻¹⁴` cutoff, same 20 steps, with the depolarizing channel
swapped for each of the others the bindings expose:

| channel | ⟨Z₆₂⟩ | final terms | peak terms | max weight | wall |
|---|---|---|---|---|---|
| `depolarize(1e-2)` | +0.002432260559 | 408 | 869 299 | 13 | 20.2 s |
| `amplitude_damping(1e-2)` | +0.125230221882 | 3 096 | 2 660 503 | 13 | 103.1 s |
| `pauli_channel(0.002, 0.002, 0.008)` | +0.002862382863 | 516 | 673 565 | 13 | 16.1 s |
| `dephase(1e-2)` | +0.005934556923 | 1 976 | 1 198 531 | 13 | 32.5 s |

All four collapse the tracked set by one to two orders of magnitude against the
`p = 0` baseline's 1.44·10⁷ peak, a property of Pauli-basis noise generally, not of
depolarizing noise specifically. `amplitude_damping` moves `⟨Z₆₂⟩` upward, to
+0.125, since it relaxes toward `|0⟩`, the `+1` eigenstate of `Z`, and is the only
channel here neither self-adjoint nor key-preserving, its Heisenberg dual
`Z → (1−γ)Z + γI` fanning out — checked against the comparisons page's
[cross-engine noise-channel parity](../comparisons.md#noise-channel-parity) results.
`dephase(1e-2)` damps `⟨Z₆₂⟩` less since dephasing leaves `Z` untouched.

## What is claimable

![Observable decay against the noise rate](../assets/b2/observable-decay-vs-noise.svg)

At `p = 0` the noise model must vanish and this showcase must return Benchmark C's
claimable numbers, read from C's committed `summary.json` at run time rather than
transcribed: the two 5-step configurations reproduce C's exact causal-cone
references (a 19-qubit reduction cross-checked between an Aer statevector and an
untruncated Pauli propagation) to gaps of 3.9·10⁻¹⁵ and 2.2·10⁻¹⁶ against a 0.01 bar,
and the 20-step configuration reproduces C's self-converged 20-step run to a gap of
exactly 0.0. Against the exact 5-step reference, `⟨Z₆₂⟩` departs from it linearly in
`p` over two decades (2.2e-16 at `p = 0`, 1.1e-5 at `1e-6`, 1.1e-3 at `1e-4`) while
the term count holds at 2 146 424 throughout, since at 5 steps the sum has already
saturated.

The circuit is ingested, not reconstructed: the 144-edge Eagle coupling map comes
from a checked-in file generated from `qiskit-ibm-runtime`'s `FakeSherbrooke`, and
`Z_62` is loaded through a provenance-tagged JSON of published operator supports.
The noiseless configuration is claimable at `θ_h = 5π/16`, 20 steps
(`⟨Z₆₂⟩ = +0.016131374386`, converged from `2⁻⁸` to `2⁻¹⁴` with a last difference of
6.50·10⁻⁴, agreeing with Benchmark C's independent reference to 0.0), and so are the
noisy configurations whose sweeps plateaued, the ones a hardware comparison would
actually want, since a device is noisy:

| configuration | ⟨Z₆₂⟩ | uncertainty | plateau at |
|---|---|---|---|
| `p = 5e-3`, 20 steps, 5π/16 | +0.006280876086 | 3.6e-4 | `2⁻¹⁴` |
| `p = 1e-2`, 20 steps, 5π/16 | +0.002483689686 | 5.1e-5 | `2⁻¹⁶` |
| `p = 3e-2`, 20 steps, 5π/16 | +0.000077952653 | 4.2e-7 | `2⁻²⁰` |

`p = 1e-3` at 20 steps is not claimable, since its cutoff sweep never plateaued, and
neither is `θ_h = 7π/32` at 20 steps noiseless: it is
[Benchmark C's measured reachability boundary](../benchmarks/c-deep-trotter.md#what-it-would-take).

### The reachability boundary, with and without noise

Running that 7π/32, 20-step circuit noisily at `p = 1e-2` shows what noise does to
the boundary. At the same `2⁻¹⁶` cutoff, the noiseless run (Benchmark C) peaks at
45.4 million terms and ends at 38.8 million, last difference 1.44e-1; the noisy run
peaks at 0.94 million, ends at 74 717, last difference 9.32e-4 — 48× fewer peak
terms, 519× fewer resident at the end, 154× smaller last difference. It still does
not pass, since the second-to-last difference is 1.08·10⁻³, over the `1e-3` bar, so
`converged = false`. `⟨Z₆₂⟩` under the noisy channel is also a different quantity
from the unitary-circuit value, agreeing only as `p → 0`, but it is the quantity a
hardware verification claim actually needs.

## Performance

Peak resident terms range from 14 396 463 at `p = 0` down to 22 105 at `p = 3e-2`,
and wall time from 531.0 s down to 0.49 s. Process RSS peaks at 1.25 GiB across the
full driver run, which covers 31 records (the noise grid, the channel variants, the
noiseless-limit and reachability legs, and the convergence sweeps) in 25.7 min
single-threaded end to end, on ccqlin038, rustc 1.94.0, Python 3.11.11.

## Validation: an independent dense noisy reference

Every oracle in the suite refuses a noise channel by construction (the statevector
oracle because it is unitary-only, the stim oracle because a tableau simulation
samples one Pauli error rather than averaging over them), so B2's correctness gate
hand-rolls a dense reference instead: gates applied to an explicit `2ⁿ × 2ⁿ` density
matrix, forward from `|0…0⟩⟨0…0|`, unitaries by conjugation and channels as
`Σ_k K_k ρ K_k†` with Kraus operators from each channel's definition, not the
engine's dual. Two-qubit Cliffords are sums of embedded Pauli products
(`CNOT = (I + Z_c + X_t − Z_c X_t)/2`, `SWAP = (II + XX + YY + ZZ)/2`), and with no
noise gate the reference must reproduce the engine's exact untruncated propagation.
At `1e-10`, for all five channels mixed with `h`/`s`/`cnot`/`cz`/`swap`, non-Clifford
`rx`/`ry`/`rz`, and a weight-two `XY` rotation:

| direction | what is compared | size |
|---|---|---|
| `heisenberg` | `⟨0…0\|Φ†(O)\|0…0⟩` from the engine against `Tr(O Φ(ρ₀))` from the Kraus evolution | `n = 6` |
| `forward` | every one of the `4ⁿ` Pauli coefficients of the engine's evolved `ρ` | `n = 4` |

29 tests, ~1 s, 97 MiB peak RSS, numpy-only.

## Caveats

Wall times are indicative, not campaign-grade: single-threaded on a shared
workstation whose stated single-thread noise is ±5–8%, run with `warmup=False`;
term counts, expectation values, and convergence verdicts are the load-independent
numbers to quote. A truncated Pauli sum has no variational bound, since discarded
terms carry signs and the error need not shrink monotonically with the cutoff;
`converged = false` means no usable uncertainty estimate, not a slightly weaker one.
The noise model is a model, not a device: a real Eagle device has different error
rates, coherent errors, crosstalk, and readout error, none of which is here. Peak,
not final, term count is the cost, since the sum grows mid-circuit and collapses by
the end.

**Numbers:** raw records in
[`examples/b2_noisy_verification/results.json`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b2_noisy_verification/results.json),
verdicts in `summary.json` next to it, source and provenance in
[`examples/b2_noisy_verification/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b2_noisy_verification/README.md).
