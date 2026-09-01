# B2 — Noise makes it cheaper

<p class="lead">Take the 127-qubit heavy-hex kicked-Ising circuit of
<a href="../benchmarks/c-deep-trotter.html">Benchmark C</a>, add per-gate noise,
and ask what the noise costs. The answer runs the opposite way to every
density-matrix method: <strong>noise makes this simulation cheaper</strong> —
651× fewer peak terms and 1078× less wall time at <code>p = 3e-2</code> than at
<code>p = 0</code>, at the same cutoff.</p>

![Peak and final term count, and wall time, against the noise rate](../assets/b2/terms-and-time-vs-noise.svg)

*The headline: peak/final resident term count and wall time against the
depolarizing rate `p`, at a fixed `min_abs_coeff = 2⁻¹⁴`, 20 Trotter steps,
single-threaded.*

## Why the tracked set collapses

A Kraus/density-matrix simulator pays for noise twice: it carries a `4ⁿ` object
instead of a `2ⁿ` one, and each channel is more work than a gate. Pauli
propagation pays *negatively*. In the Pauli basis a single-qubit depolarizing
channel is a pure coefficient rescale — `1 − 4p/3` per non-identity qubit on its
support — so a weight-`w` string crossing one full noise layer loses
`(1 − 4p/3)^w`, and after `d` layers a string that has held weight `w` throughout
is down by `(1 − 4p/3)^{wd}`. Two things follow:

- the filter is **exponential in weight**, so a fixed `min_abs_coeff` imposes an
  effective weight cap that tightens with depth — and the tracked set of a
  scrambling circuit is dominated by its high-weight tail;
- nothing about it is heuristic. At the Clifford kick angle `θ_h = π/2` the
  evolved operator stays a *single* Pauli string whose coefficient is exactly
  `(1 − 4p/3)^hits`, with `hits` the number of channels whose qubit was in the
  string's support at the time. The CI gate counts those hits by hand and checks
  the product to `1e-12` relative.

The noise model is stated exactly, because every number depends on it. One
single-qubit noise channel is pushed on **every qubit in the support of the gate
that just ran**, so a qubit takes `deg(q) + 1` channels per Trotter step — 2 to 4
on the Eagle lattice. That is `2n + 3|E| = 686` channels per step at `n = 127`
against 271 for the noiseless circuit, i.e. **13 720** channels at 20 steps. One
gate per channel throughout, so the noise channels are *also* truncation points
— which is the entire mechanism.

`p = 0` is run as `depolarize(0.0)` channels rather than as a different circuit.
`1 − 4·0/3` is exactly `1.0`, the rescale is exactly key-preserving, and
re-truncating an unchanged sum drops nothing new, so the `p = 0` leg is the
*same computation* as the noiseless one while keeping the noisy legs' channel
schedule. A test pins that term for term.

## The collapse, measured

`θ_h = 5π/16`, 20 Trotter steps, `Z_62`, Heisenberg, contracted against `|0…0⟩`,
single-threaded, fixed `min_abs_coeff = 2⁻¹⁴`:

| p | ⟨Z₆₂⟩ | final terms | peak terms | max weight | mean weight | wall (1 thread) | peak vs `p=0` |
|---|---|---|---|---|---|---|---|
| **0** | +0.016131374386 | 34 698 | **14 396 463** | 19 | 10.30 | **531.0 s** | — |
| 1e-3 | +0.012998050638 | 20 098 | 9 710 246 | 19 | 9.80 | 361.1 s | 1.5× fewer |
| 5e-3 | +0.006280876086 | 3 521 | 2 818 675 | 14 | 8.61 | 82.5 s | 5.1× fewer |
| 1e-2 | +0.002432260559 | 408 | 869 299 | 13 | 7.41 | 20.2 s | 16.6× fewer |
| **3e-2** | +0.000000000000 | **0** | **22 105** | 0 | 0 | **0.49 s** | **651× fewer** |

Three things to read off:

- **The collapse is in the weight distribution, not just the count.** Maximum
  surviving Pauli weight goes 19, 19, 14, 13, 0 and the mean 10.3, 9.8, 8.6,
  7.4, 0 — the exponential-in-weight filter showing up directly.
- **Cost tracks the peak, and the peak is the collapse.** Wall time falls faster
  than the term count (1078× against 651×): per-layer cost is not linear in
  resident terms, since the merge phase carries a sort and a multi-million-term
  sum is far out of cache, so shrinking the tracked set pays twice.
- **The observable decays too, and that is physics, not truncation.** A
  depolarizing channel is a contraction; after 13 720 of them the signal is
  small. Which is exactly why the fixed-cutoff column cannot be read on its own.

The honest comparison here is *within this engine*: `p = 0` in the same sweep. A
density-matrix method at `n = 127` is not on the table at all (`4^127`
amplitudes). The claim is not "Pauli propagation beats a density-matrix
simulator on noisy circuits"; it is that **adding noise to a Pauli-propagation
run makes that run cheaper**, which is the opposite of how every Kraus-based
method scales.

## The convergence panel, and the column that reads zero

![Convergence against cutoff, one curve per noise rate](../assets/b2/convergence-vs-cutoff.svg)

This is the showcase where the convergence requirement earns its keep: **at
`p = 3e-2` the fixed-cutoff answer is exactly `0`, and the true answer is
7.8·10⁻⁵.**

| p | 2⁻⁸ | 2⁻¹⁰ | 2⁻¹² | 2⁻¹⁴ | 2⁻¹⁶ | 2⁻¹⁸ | 2⁻²⁰ | last Δ | plateau? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.011941 | 0.014730 | 0.015481 | **0.016131** | — | — | — | 6.50e-4 | **yes** |
| 1e-3 | 0.007101 | 0.011707 | 0.012737 | **0.012998** | — | — | — | 2.61e-4 | **no** |
| 5e-3 | 0 | 0.005486 | 0.005918 | **0.006281** | — | — | — | 3.63e-4 | **yes** |
| 1e-2 | 0 | 0 | 0.002272 | 0.002432 | **0.002484** | — | — | 5.14e-5 | **yes** |
| 3e-2 | 0 | 0 | 0 | **0** | 7.156e-5 | 7.753e-5 | **7.795e-5** | 4.23e-7 | **yes** |

- **A fixed cutoff is not a fixed accuracy.** As `p` grows the signal shrinks, so
  the cutoff has to be tightened *along with* `p`. At `p = 3e-2` a `2⁻¹⁴` cutoff
  sits above every contributing coefficient, the tracked set empties, and the
  contraction returns `0` from an empty sum.
- **Noise buys *cutoff reach*, which is what converts into accuracy.** The whole
  `p = 3e-2` sweep out to `2⁻²⁰` costs 76 s — an eighth of the single `p = 0`
  point at `2⁻¹⁴` — and resolves the answer to **4.2·10⁻⁷** against
  **6.5·10⁻⁴** at `p = 0`. That is 1500× smaller uncertainty for an eighth of
  the cost. At `p = 0` the next dyadic pair is simply out of reach: that leg's
  own measured growth projects ~1.9·10⁸ terms, ~9 GiB of columns and ~2 h
  single-threaded.
- **The verdict is Benchmark B's plateau criterion, imported as a function
  object** — not re-implemented. It requires the last *two* successive
  differences below `1e-3` **and** either a saturated term count or two
  strictly-nonzero differences, and it rejects an empty sum outright.
- **It rejects `p = 1e-3`,** whose last difference is a comfortable 2.61·10⁻⁴ but
  whose previous one is 1.03·10⁻³, just over tolerance. That leg is reported as
  unresolved rather than quoted — which is why the story is "noise helps, and
  here is where it has not helped *enough* yet" rather than a clean
  five-for-five.
- **Resolution improves with `p` in absolute terms, not relative ones.** The last
  difference falls 6.50e-4 → 3.63e-4 → 5.14e-5 → 4.23e-7 across the resolved
  legs, but as a fraction of the signal it is 4.0%, 5.8%, 2.1%, 0.54% — better
  overall, not monotone. The bar is absolute, so the absolute column is the
  relevant one; the relative one is stated so nobody reads more into it.

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
`p = 0` baseline's 1.44·10⁷ peak, so the mechanism is a property of Pauli-basis
noise, not of depolarizing noise specifically. Two differences are worth naming
rather than averaging over:

- **`amplitude_damping` moves `⟨Z₆₂⟩` the wrong way — upwards, to +0.125.** That
  is correct: the channel relaxes towards `|0⟩`, the `+1` eigenstate of `Z`, so
  it *drives* `⟨Z⟩` towards `+1`. It is also the only channel here that is
  neither self-adjoint nor key-preserving — its Heisenberg dual `Z → (1−γ)Z + γI`
  fans out — so the engine takes the gather/sort/merge path rather than the
  in-place rescale. The comparison is not controlled (`γ` and a depolarizing `p`
  are not the same knob), and this is the channel whose `apply`/`apply_adjoint`
  orientation was wrong until it was caught by the
  [cross-engine baseline](../comparisons.md#a-real-bug-the-baseline-caught), so
  it gets its own CI checks.
- **`dephase(1e-2)` damps less than `depolarize(1e-2)` on this observable.**
  Dephasing leaves `Z` untouched and scales only `X`/`Y`, and `⟨Z₆₂⟩` is read
  off the `I`/`Z`-only strings, so it suppresses the branches that *feed* the
  answer rather than the answer itself.

These are single runs at one cutoff, not converged answers — they are here to
show the collapse is general. Only the depolarizing legs carry a convergence
sweep.

## The noiseless limit recovers Benchmark C

At `p = 0` the noise model must vanish and this showcase must return Benchmark
C's numbers. It is scored against **only** C's `claimable` rows, read out of C's
committed `summary.json` at run time rather than transcribed — so a change to C's
labels breaks this check instead of silently invalidating it.

| θ_h | steps | this showcase, `p = 0` | Benchmark C | C's method | gap | bar |
|---|---|---|---|---|---|---|
| 7π/32 | 5 | +0.655563050749 (`2⁻¹⁸`) | +0.655563050749 | `light_cone_exact:both`, **exact** | **3.9e-15** | 0.01 |
| 5π/16 | 5 | +0.238477118019 (`2⁻¹⁸`) | +0.238477118019 | `light_cone_exact:both`, **exact** | **2.2e-16** | 0.01 |
| 5π/16 | 20 | +0.016131374386 (`2⁻¹⁴`) | +0.016131374386 | self-converged, plateau at `2⁻¹⁴` | **0.0** | 7.5e-4 |

The two 5-step rows carry the weight: their references are *exact* — a 19-qubit
commutation-aware causal cone evaluated twice, by an Aer statevector and by an
untruncated Pauli propagation, required to agree — so reproducing them to
4·10⁻¹⁵ says the whole noisy construction at `p = 0` (lattice, coloured edge
order, layer order, rotation convention, the 415 extra no-op channels per step)
is the same physics C measured. The 20-step row is a **reproducibility** check,
not an accuracy one: C's 20-step reference is C's own tightest run, so a zero gap
means the added `depolarize(0.0)` channels did not perturb the computation.

Taking the limit properly, against the exact 5-step reference:

| p | ⟨Z₆₂⟩ | \|gap vs exact\| | final terms |
|---|---|---|---|
| 0 | +0.238477118019 | 2.2e-16 | 2 146 424 |
| 1e-6 | +0.238466081368 | 1.1e-5 | 2 146 424 |
| 1e-4 | +0.237376185354 | 1.1e-3 | 2 146 424 |

Linear in `p` over two decades, as a first-order channel expansion requires, and
the term count is unchanged — at 5 steps and this cutoff the sum has saturated,
so weak noise moves coefficients without moving the tracked set. **The collapse
is a deep-circuit effect:** it needs enough noise layers for `(1 − 4p/3)^{wd}` to
cross the cutoff.

## What is claimable, and what is not

![Observable decay against the noise rate](../assets/b2/observable-decay-vs-noise.svg)

The circuit is *ingested* rather than reconstructed: the 144-edge Eagle coupling
map comes from a checked-in file generated from `qiskit-ibm-runtime`'s
`FakeSherbrooke`, never hand-typed, and the observable `Z_62` is loaded through a
provenance-tagged JSON of published operator supports.

**Claimable — the noiseless configuration.** `θ_h = 5π/16`, 20 steps:

> `⟨Z₆₂⟩ = +0.016131374386`, with this showcase's own convergence evidence
> (0.011941 → 0.014730 → 0.015481 → 0.016131 across `2⁻⁸…2⁻¹⁴`, last two
> differences 7.52·10⁻⁴ and 6.50·10⁻⁴, plateau accepted), agreeing with
> Benchmark C's independently-recorded claimable reference to 0.0.

**Claimable — the noisy configurations whose sweeps plateaued.** These are the
ones a *hardware* comparison would actually want, since a device is noisy:

| configuration | ⟨Z₆₂⟩ | uncertainty | plateau at |
|---|---|---|---|
| `p = 5e-3`, 20 steps, 5π/16 | +0.006280876086 | 3.6e-4 | `2⁻¹⁴` |
| `p = 1e-2`, 20 steps, 5π/16 | +0.002483689686 | 5.1e-5 | `2⁻¹⁶` |
| `p = 3e-2`, 20 steps, 5π/16 | +0.000077952653 | 4.2e-7 | `2⁻²⁰` |

**Not claimable, and why.** `p = 1e-3` at 20 steps — its cutoff sweep never
plateaued; the value it reports, 0.012998, is probably close, and "probably
close" is not a claim. And `θ_h = 7π/32` at 20 steps noiseless, which is
[Benchmark C's measured reachability boundary](../benchmarks/c-deep-trotter.md#what-it-would-take):
nothing in B2 improves the *noiseless* answer there.

### The reachability boundary, with and without noise

The interesting question is what noise does to that boundary, so the same 7π/32,
20-step circuit was run at `p = 1e-2`. At the **same** `2⁻¹⁶` cutoff:

| | peak terms | still resident at the end | last Δ |
|---|---|---|---|
| noiseless (Benchmark C) | 45.4 million | 38.8 million | 1.44e-1 |
| `p = 1e-2` (this run) | 0.94 million | 74 717 | 9.32e-4 |

**48× fewer peak terms, 519× fewer still resident at the end, and a 154× smaller
last difference.** And it *still* does not pass: the plateau test needs the last
two differences under `1e-3` and the second-to-last is 1.08·10⁻³, so the verdict
stays `converged = false` and the value stays unclaimable. The criterion is not
bent to fit the result.

Two caveats, both load-bearing:

- **This is a different quantity.** `⟨Z₆₂⟩` under the noisy channel is not
  `⟨Z₆₂⟩` under the unitary circuit; they agree only as `p → 0`, and at
  `p = 1e-2` after 20 steps they are far apart (0.1095 against C's unresolved
  ~0.25–0.52 band). **Noise is not a cheaper route to the noiseless number.**
- **But it is the physically relevant one for a verification claim.** A device
  running this circuit has noise; the quantity a classical simulation should be
  asked to reproduce is the noisy one. That regime is precisely the cheap one
  here — which is the practical form of the headline.

## Validation: an independent dense noisy reference

Every oracle in the suite **refuses** a noise channel by construction — the
statevector oracle because it is unitary-only, the stim oracle because a tableau
simulation samples one Pauli error rather than averaging over them — and both
say so, pointing at "a density-matrix reference, which this module does not
provide". So B2's correctness gate provides one, hand-rolled:

- the circuit's gates are applied to an explicit `2ⁿ × 2ⁿ` density matrix,
  forward in time from `|0…0⟩⟨0…0|`, unitaries by conjugation and channels as
  `Σ_k K_k ρ K_k†` with the Kraus operators written from each channel's
  *definition* — not from the engine's dual;
- two-qubit Cliffords are expressed as sums of embedded Pauli products
  (`CNOT = (I + Z_c + X_t − Z_c X_t)/2`, `SWAP = (II + XX + YY + ZZ)/2`), so the
  reference never has to reason about tensor-factor ordering for a non-adjacent
  pair — the one place a dense cross-check is easy to get silently wrong;
- the reference is itself guarded: with no noise gate it must reproduce the
  engine's exact untruncated Pauli propagation.

What it checks, at `1e-10`, for all five channels mixed with `h`/`s`/`cnot`/
`cz`/`swap`, non-Clifford `rx`/`ry`/`rz` and a weight-two `XY` rotation:

| direction | what is compared | size |
|---|---|---|
| `heisenberg` | `⟨0…0\|Φ†(O)\|0…0⟩` from the engine against `Tr(O Φ(ρ₀))` from the Kraus evolution | `n = 6` |
| `forward` | **every one of the `4ⁿ` Pauli coefficients** of the engine's evolved `ρ`, so a *missing* term fails too | `n = 4` |

29 tests, ~1 s, 97 MiB peak RSS, numpy-only, no `importorskip`.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py           # ~26 min
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick   # ~1 s, 20 qubits, writes nothing
python examples/b2_noisy_verification/run_b2.py --figures-only                # re-render the SVGs
pytest python/paulistrings/tests/test_showcase_b2.py                          # the CI gate
```

`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts; the
driver refuses to run otherwise.

## Caveats and sources

- **Wall times are indicative, not campaign-grade.** Single-threaded on a shared
  workstation whose stated single-thread noise is ±5–8%, and taken with
  `warmup=False` (the tight-cutoff `p = 0` leg costs minutes, so a discarded
  warm-up pass would double the whole run). Term counts, expectation values and
  convergence outcomes are load-independent and are the numbers to quote.
- **A truncated Pauli sum has no variational bound.** Discarded terms carry
  signs, so a partial sum can sit on either side of the truth and the error need
  not be monotone in the cutoff. That is why every `p` carries a cutoff sweep and
  a plateau verdict rather than a single number, and why `converged = false` must
  be read as "no usable uncertainty estimate", not "a slightly weaker one".
- **The noise model is a model, not a device.** One depolarizing channel per
  gate-qubit at a single rate is the standard first-order stand-in; a real Eagle
  device has different one- and two-qubit error rates, coherent errors, crosstalk
  and readout error, none of which is here. `p = 1e-3` is the order of a good
  two-qubit gate error; `1e-2` and `3e-2` are deliberately past it, chosen to
  show the collapse rather than to model hardware.
- **Peak, not final, is the cost.** At the noisier settings the two differ by
  orders of magnitude — the sum blows up mid-circuit and collapses by the end —
  and it is the peak a run has to hold.
- Recorded run: ccqlin038 (Intel Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0,
  Python 3.11.11, single-threaded, 25.7 min end to end, 31 records.

**Source for every number on this page:**
[`examples/b2_noisy_verification/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b2_noisy_verification/README.md),
with the raw records in `results.json` and the verdicts in `summary.json` next to
it.
