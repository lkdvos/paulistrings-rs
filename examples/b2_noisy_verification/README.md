# Showcase B2 — noisy simulation, and what it verifies

Part B entry **B2** of [`research/plans/2026-08-31-examples-benchmarks-suite.md`](../../research/plans/2026-08-31-examples-benchmarks-suite.md)
(§6 Part B). The 127-qubit heavy-hex kicked Ising circuit of Benchmark C — `θ_zz = −π/2`,
observable `Z_62`, Heisenberg picture, contracted against `|0…0⟩`, one gate per channel — with
**per-gate noise added**, and the question: what does noise cost?

The answer is the point of this showcase, and it runs the opposite way to every density-matrix
method: **noise makes this simulation cheaper.** A Kraus/density-matrix simulator pays for noise
twice — it carries a `4^n` object instead of a `2^n` one, and each channel is more work than a gate.
Pauli propagation pays *negatively*: in the Pauli basis a single-qubit depolarizing channel is a
pure coefficient rescale, `1 − 4p/3` per non-identity qubit on its support, so a fixed
`min_abs_coeff` becomes a weight- and depth-dependent filter that sharpens as `p` grows. The
tracked set collapses.

| file | what it is |
|---|---|
| [`run_b2.py`](run_b2.py) | the driver — five parts, and the record of every measurement that shaped it (`MEASURED_PILOT`, `COEFF_GRID_CUTS`) |
| `results.json` | every `report.RunRecord` with full provenance |
| `summary.json` | the sweep legs with their convergence verdicts, the cited Benchmark C rows, the cuts, the pilot |
| `terms-and-time-vs-noise.svg` | **the headline**: peak/final term count and wall time against `p`, at a fixed cutoff |
| `observable-decay-vs-noise.svg` | `⟨Z₆₂⟩` against `p`, fixed cutoff vs tightest affordable cutoff |
| `convergence-vs-cutoff.svg` | plan §7 rule 4's convergence panel, one curve per `p` — and the cutoff reach the collapse buys |

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick  # ~1 s, 20q, writes nothing
python examples/b2_noisy_verification/run_b2.py --figures-only              # re-render the SVGs
```

`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts (Rayon builds its global
pool at the first propagate and never resizes it); the driver refuses to run otherwise. The CI-safe
correctness gate is [`python/paulistrings/tests/test_showcase_b2.py`](../../python/paulistrings/tests/test_showcase_b2.py)
(29 tests, ~1 s, 97 MiB peak RSS, numpy-only) — see §6.

## 0. The short version

Recorded run: commit `a3d260f` (worktree dirty — see §8), ccqlin038 (Intel Xeon Gold 6244 @
3.60 GHz), rustc 1.94.0, python 3.11.11, paulistrings 0.1.0, single-threaded, **25.7 min** end to end, 31
`RunRecord`s. Four results, in order of how much they should change what a reader believes:

1. **At a fixed cutoff, noise collapses the tracked set — monotonically, and by orders of
   magnitude.** At `min_abs_coeff = 2⁻¹⁴`, 20 Trotter steps, `θ_h = 5π/16`: peak resident terms fall
   from **14 396 463** at `p = 0` to **22 105** at `p = 3e-2` (**651×**), wall time from **531 s** to
   **0.49 s** (**1078×**), and the maximum surviving Pauli weight from **19** to **0**. Nothing in
   that is a fitted trend: the mechanism is an exact per-channel factor (§1.1), and the CI gate
   checks it to `1e-12` relative at the Clifford point.
2. **Noise also buys *cutoff reach*, which is what actually converts into accuracy.** The tightest
   affordable cutoff moves `2⁻¹⁴ → 2⁻¹⁴ → 2⁻¹⁴ → 2⁻¹⁶ → 2⁻²⁰` across the `p` grid, and the resolved
   uncertainty with it: **6.5e-4** at `p = 0` (572 s for that sweep) against **4.2e-7** at `p = 3e-2`
   (76 s for its whole sweep) — 1500× smaller for an eighth of the cost. At `p = 0` the next
   dyadic pair is simply out of reach: that leg's own measured growth (13.4× in peak terms and
   ~14× in wall time per dyadic pair) projects ~1.9e8 terms, ~9 GiB of columns and ~2 h
   single-threaded.
3. **The noiseless limit reproduces Benchmark C exactly.** At `p = 0` the 20-step run returns
   `+0.016131374386` with 34 698 final and 14 396 463 peak terms — C's committed reference value and
   C's committed term counts, digit for digit (gap **0.0**) — and the two 5-step legs reproduce C's
   *exact* causal-cone references to **3.9e-15** and **2.2e-16**. Taking `p → 0` on an exact 5-step
   reference gives a clean linear departure: 2.2e-16 at `p = 0`, 1.1e-5 at `1e-6`, 1.1e-3 at `1e-4`.
4. **What is claimable is narrower than what is computable, and the driver says which is which.**
   Claimable: `θ_h = 5π/16`, 20 steps, noiseless (against C's claimable row), and the `p ∈ {5e-3,
   1e-2, 3e-2}` legs, whose cutoff sweeps plateau. **Not** claimable: `p = 1e-3` (its sweep never
   plateaued — 2.6e-4 last difference but 1.03e-3 the one before), and `θ_h = 7π/32` at 20 steps
   noiseless, which is C's measured reachability boundary. Adding `p = 1e-2` to that 7π/32 point cuts
   the last difference from C's 1.44e-1 at 4.5e7 peak terms to 9.3e-4 at 9.4e5 — 48× fewer terms,
   154× better — and *still* does not pass the plateau test, and would answer a different question
   anyway (§5.1).

## 1. The circuit, and the noise model stated exactly

The unitary part is Benchmark C's circuit, built by the same shared builder
(`examples/common/circuits.py::heavy_hex_kicked_ising`): the 144-edge Eagle r3 heavy-hex lattice
read from the *generated* [`examples/data/heavy_hex_127.edges`](../data/heavy_hex_127.edges), ZZ
rotations emitted in the 3-colored (disjoint-support) hardware order, layer order `x-then-zz`,
`θ_zz = −π/2`, kick angle `θ_h` from C's hard interior. One Trotter step is

```
for q in 0..n-1:            rx(θ_h, q)                     then  N(q)
for (a, b) in colored E:    pauli_rotation("ZZ", ab, θ_zz)  then  N(a), N(b)
```

where **`N` is one single-qubit noise channel on every qubit in the support of the gate that just
ran**, pushed immediately after it. Consequences worth stating, because every number below depends
on them:

* A qubit `q` takes `deg(q) + 1` channels per Trotter step — 2 to 4 on the Eagle lattice
  (degrees 1–3). It is a per-*gate* model, not a per-*layer* one.
* Channels per step are `2n + 3|E|` = **686** at `n = 127`, against 271 for the noiseless circuit;
  **13 720** channels at 20 steps. One gate per channel throughout (plan §5, decision D10), so the
  noise channels are *also* truncation points — which is the entire mechanism.
* `p = 0` is run as `depolarize(0.0)` channels, not as a different circuit. `1 − 4·0/3` is exactly
  `1.0`, the rescale is exactly key-preserving, and re-truncating an unchanged sum drops nothing
  new, so the `p = 0` leg is the *same computation* as the noiseless one while keeping the noisy
  legs' channel schedule.
  `test_showcase_b2.py::test_p_zero_leg_matches_the_noiseless_circuit` pins that term for term, and
  `test_noiseless_model_reproduces_the_shared_builder` pins that the noiseless model is the shared
  builder's circuit.

The other three channels in §3 slot into the same `N`.

### 1.1 Why the tracked set collapses

Each `depolarize(p, q)` multiplies a Pauli string's coefficient by `1 − 4p/3` exactly when the
string is non-identity on `q`, and by `1` otherwise (`crates/paulistrings/src/channel/noise.rs`).
So a weight-`w` string crossing one full noise layer loses `(1 − 4p/3)^w`, and after `d` layers a
string that has held weight `w` throughout is down by `(1 − 4p/3)^{wd}`. Two things follow:

* the filter is **exponential in weight**, so a fixed `min_abs_coeff` imposes an effective weight
  cap that tightens with depth — and the tracked set of a scrambling circuit is dominated by its
  high-weight tail;
* nothing about it is heuristic. At the Clifford kick angle `θ_h = π/2` the evolved operator stays a
  *single* Pauli string, and its coefficient is exactly `(1 − 4p/3)^hits` with `hits` the number of
  channels whose qubit was in the string's support at the time.
  `test_clifford_point_coefficient_is_exactly_the_hand_counted_decay` counts those hits by hand and
  checks the product to `1e-12` relative.

## 2. Noise accelerates truncation

`θ_h = 5π/16`, 20 Trotter steps, `Z_62`, Heisenberg, `|0…0⟩`, single-threaded, one fixed cutoff
`min_abs_coeff = 2⁻¹⁴` — Benchmark C's middle dyadic, the one whose 20-step tracked set sits dead
centre of the handoff's 1.2e6–9.3e6 envelope. This is `terms-and-time-vs-noise.svg`:

| p | ⟨Z₆₂⟩ | final terms | peak terms | max weight | mean weight | wall (1 thread) | peak vs `p=0` |
|---|---|---|---|---|---|---|---|
| **0** | +0.016131374386 | 34 698 | **14 396 463** | 19 | 10.30 | **531.0 s** | — |
| 1e-3 | +0.012998050638 | 20 098 | 9 710 246 | 19 | 9.80 | 361.1 s | 1.5× fewer |
| 5e-3 | +0.006280876086 | 3 521 | 2 818 675 | 14 | 8.61 | 82.5 s | 5.1× fewer |
| 1e-2 | +0.002432260559 | 408 | 869 299 | 13 | 7.41 | 20.2 s | 16.6× fewer |
| **3e-2** | +0.000000000000 | **0** | **22 105** | 0 | 0 | **0.49 s** | **651× fewer** |

Three things to read off it:

* **The collapse is in the weight distribution, not just the count.** The maximum surviving Pauli
  weight goes 19, 19, 14, 13, 0 and the mean 10.3, 9.8, 8.6, 7.4, 0. That is §1.1's prediction
  showing up directly: the noise factor is exponential in weight, so a fixed coefficient cutoff acts
  as an effective weight cap, and it is the high-weight tail that carries the term count of a
  scrambling circuit.
* **Cost tracks the peak, and the peak is the collapse.** Wall time falls faster than the term count
  (1078× against 651×): per-layer cost is not linear in resident terms — the merge phase carries a
  sort, and a multi-million-term sum is far out of cache — so shrinking the tracked set pays twice.
* **The observable decays too, and that is physics, not truncation.** `⟨Z₆₂⟩` falls 0.0161 → 7.8e-5
  across the grid (`observable-decay-vs-noise.svg`). A depolarizing channel is a contraction; after
  13 720 of them the signal is small. Which is precisely why the fixed-cutoff column cannot be read
  on its own — see §2.1.

The comparison that matters here is *within this engine*: `p = 0` in the same sweep is the honest
baseline, since a density-matrix method at `n = 127` is not on the table at all (`4^127` amplitudes).
The claim is not "Pauli propagation beats a density-matrix simulator on noisy circuits"; it is that
**adding noise to a Pauli-propagation run makes that run cheaper**, which is the opposite of how
every Kraus-based method scales.

### 2.1 The convergence panel, and the `p = 3e-2` column that reads zero

Plan §7 rule 4 requires a convergence panel on every truncated result, and this is the showcase
where it earns its keep: **at `p = 3e-2` the fixed-cutoff answer is exactly `0`, and the true answer
is 7.8e-5.** `convergence-vs-cutoff.svg` (left panel) is the whole dyadic sweep behind every point in
the table above:

| p | 2⁻⁸ | 2⁻¹⁰ | 2⁻¹² | 2⁻¹⁴ | 2⁻¹⁶ | 2⁻¹⁸ | 2⁻²⁰ | last Δ | plateau? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.011941 | 0.014730 | 0.015481 | **0.016131** | — | — | — | 6.50e-4 | **yes** |
| 1e-3 | 0.007101 | 0.011707 | 0.012737 | **0.012998** | — | — | — | 2.61e-4 | **no** |
| 5e-3 | 0 | 0.005486 | 0.005918 | **0.006281** | — | — | — | 3.63e-4 | **yes** |
| 1e-2 | 0 | 0 | 0.002272 | 0.002432 | **0.002484** | — | — | 5.14e-5 | **yes** |
| 3e-2 | 0 | 0 | 0 | **0** | 7.156e-5 | 7.753e-5 | **7.795e-5** | 4.23e-7 | **yes** |

* **A fixed cutoff is not a fixed accuracy.** As `p` grows the signal shrinks, so the cutoff has to
  be tightened *along with* `p` — and at `p = 3e-2` a `2⁻¹⁴` cutoff is above every coefficient that
  contributes, so the tracked set empties and the contraction returns `0` from an empty sum. The
  point of the collapse is that tightening is then affordable: the whole `p = 3e-2` sweep out to
  `2⁻²⁰` costs 76 s (60.3 s of that the `2⁻²⁰` point alone), an eighth of the single `p = 0` point
  at `2⁻¹⁴`.
* **The verdict is Benchmark B's plateau criterion, imported as a function object**
  (`bench_b_theta_sweep._plateau_is_real` — the same reuse Benchmark C makes, asserted by
  `test_convergence_verdict_uses_benchmark_bs_plateau_criterion`). It requires the last *two*
  successive differences below `1e-3` **and** either a saturated term count or two
  strictly-nonzero differences, and it rejects an empty sum outright — B measured that the naive
  "the last two values agree" test reports an uncertainty of exactly zero on a value that is still
  wrong.
* **It rejects `p = 1e-3`,** whose last difference is a comfortable 2.61e-4 but whose previous one is
  1.03e-3, just over tolerance. That leg is reported as unresolved rather than quoted, and it is the
  reason the `p` grid's story is "noise helps, and here is where it has not helped *enough* yet"
  rather than a clean five-for-five.
* **Resolution improves with `p` in absolute terms, not in relative ones.** The last difference falls
  6.50e-4 → 3.63e-4 → 5.14e-5 → 4.23e-7 across the resolved legs, but as a fraction of the signal it
  is 4.0 %, 5.8 %, 2.1 %, 0.54 % — better overall, not monotone. The plan's bar is *absolute*
  (`|error| < 0.01`), so the absolute column is the relevant one; the relative column is stated so
  nobody reads more into it.

## 3. The same collapse, three other channels

Same circuit, same `2⁻¹⁴` cutoff, same 20 steps — with the depolarizing channel swapped for each of
the other three the bindings expose in the `N` slot of §1:

| channel | ⟨Z₆₂⟩ | final terms | peak terms | max weight | wall |
|---|---|---|---|---|---|
| `depolarize(1e-2)` *(from §2)* | +0.002432260559 | 408 | 869 299 | 13 | 20.2 s |
| `amplitude_damping(1e-2)` | +0.125230221882 | 3 096 | 2 660 503 | 13 | 103.1 s |
| `pauli_channel(0.002, 0.002, 0.008)` | +0.002862382863 | 516 | 673 565 | 13 | 16.1 s |
| `dephase(1e-2)` | +0.005934556923 | 1 976 | 1 198 531 | 13 | 32.5 s |

All four collapse the tracked set by one to two orders of magnitude against the `p = 0` baseline's
1.44e7 peak terms, so the mechanism is a property of Pauli-basis noise, not of depolarizing noise
specifically. Two differences are worth naming rather than averaging over:

* **`amplitude_damping` moves `⟨Z₆₂⟩` the wrong way — upwards, to +0.125.** That is correct: this
  channel relaxes towards `|0⟩`, which is the `+1` eigenstate of `Z`, so it *drives* `⟨Z⟩` towards
  `+1` rather than towards zero. It is also the only channel here that is neither self-adjoint nor
  key-preserving — its Heisenberg dual is `Z → (1−γ)Z + γI`, a fan-out, so the engine takes the
  gather/sort/merge path rather than the in-place rescale the other three get. It is the most
  expensive of the four here, though the comparison is not controlled: `γ` and a depolarizing `p`
  are not the same knob, and its tracked set is also the largest. This is the channel whose
  `apply`/`apply_adjoint` orientation was wrong until commit `e42095c`, so it gets its own CI
  checks (§6).
* **`dephase(1e-2)` damps less than `depolarize(1e-2)` on this observable.** Dephasing leaves `Z`
  untouched and scales only `X`/`Y` (by `1 − 2p`), and `⟨Z₆₂⟩` is read off the `I`/`Z`-only strings,
  so it suppresses the branches that *feed* the answer rather than the answer itself. The general
  `pauli_channel` here (`px = py = 0.002`, `pz = 0.008`) is deliberately asymmetric and lands close
  to `depolarize(1e-2)`, as its dual factors predict.

These are single runs at one cutoff, not converged answers — they are here to show the collapse is
general. Only §2's depolarizing legs carry a convergence sweep.

## 4. The noiseless limit recovers Benchmark C

At `p = 0` the noise model must vanish and this showcase must return Benchmark C's numbers. The rows
it is scored against are **only** C's `claimable` ones, and they are read out of
[`benchmarks/python/deep_trotter/summary.json`](../../benchmarks/python/deep_trotter/summary.json)
(commits `e024d8b` / `01a057c`) at run time by `claimable_references()`, not transcribed — a change
to C's labels breaks this check instead of silently invalidating it.
`test_claimable_references_are_exactly_benchmark_cs_claimable_rows` pins both the values and the
claimability labels from the other side.

| θ_h | steps | this showcase, `p = 0` | Benchmark C | C's method | gap | bar |
|---|---|---|---|---|---|---|
| 7π/32 | 5 | +0.655563050749 (`2⁻¹⁸`) | +0.655563050749 | `light_cone_exact:both`, **exact** | **3.9e-15** | 0.01 |
| 5π/16 | 5 | +0.238477118019 (`2⁻¹⁸`) | +0.238477118019 | `light_cone_exact:both`, **exact** | **2.2e-16** | 0.01 |
| 5π/16 | 20 | +0.016131374386 (`2⁻¹⁴`) | +0.016131374386 | self-converged, plateau at `2⁻¹⁴` | **0.0** | 7.5e-4 |

The two 5-step rows are the ones that carry weight: their references are *exact* (a 19-qubit
commutation-aware causal cone, evaluated twice — an Aer statevector and an untruncated Pauli
propagation — and required to agree), so reproducing them to `4e-15` says the whole
`noisy_kicked_ising` construction at `p = 0` — lattice, colored edge order, layer order, rotation
convention, the 415 extra no-op channels per step — is the same physics C measured.

The 20-step row is a **reproducibility** check, not an accuracy one: C's 20-step reference is
self-converged, i.e. C's own tightest run, so a zero gap means the added `depolarize(0.0)` channels
did not perturb the computation — which is exactly the claim §1 makes about them, and it comes out
bit-identical, including both term counts (34 698 final / 14 396 463 peak, matching C's envelope
table). It says nothing about whether 0.016131 is the truth; C's §2.3 is the record of how far a
self-converged uncertainty can be trusted.

Taking the limit properly, against the exact 5-step reference at `θ_h = 5π/16`, `2⁻¹⁸`:

| p | ⟨Z₆₂⟩ | \|gap vs exact\| | final terms |
|---|---|---|---|
| 0 | +0.238477118019 | 2.2e-16 | 2 146 424 |
| 1e-6 | +0.238466081368 | 1.1e-5 | 2 146 424 |
| 1e-4 | +0.237376185354 | 1.1e-3 | 2 146 424 |

Linear in `p` over two decades, as a first-order channel expansion requires, and the term count is
unchanged — at 5 steps and this cutoff the sum has saturated, so weak noise moves coefficients
without moving the tracked set. The collapse of §2 is a *deep-circuit* effect: it needs enough noise
layers for `(1 − 4p/3)^{wd}` to cross the cutoff.

## 5. What this verifies, and what it does not

The utility-verification framing, stated as plainly as the evidence allows. The circuit is *ingested*
rather than reconstructed: the 144-edge Eagle coupling map comes from the checked-in, generated
[`examples/data/heavy_hex_127.edges`](../data/heavy_hex_127.edges) (produced from `qiskit-ibm-runtime`'s
`FakeSherbrooke` by [`generate_heavy_hex.py`](../data/generate_heavy_hex.py), never hand-typed), and
the observable `Z_62` is the weight-1 operator of the experiment's Fig. 4b, loaded through the
provenance-tagged [`kim2023_observables.json`](../data/kim2023_observables.json).

What the showcase returns, and with what standing:

**Claimable — the noiseless configuration.** `θ_h = 5π/16`, 20 Trotter steps, noiseless:

> `⟨Z₆₂⟩ = +0.016131374386`, with this showcase's own convergence evidence — the cutoff sweep
> 0.011941 → 0.014730 → 0.015481 → 0.016131 across `2⁻⁸…2⁻¹⁴`, last two differences 7.52e-4 and
> 6.50e-4, plateau accepted — and agreeing with Benchmark C's independently-recorded claimable
> reference (`+0.016131374386 ± 7.5e-4`) to 0.0.

**Claimable — the noisy configurations whose sweeps plateaued.** These are the ones a *hardware*
comparison would actually want, since a device is noisy:

| configuration | ⟨Z₆₂⟩ | uncertainty | plateau at |
|---|---|---|---|
| `p = 5e-3`, 20 steps, 5π/16 | +0.006280876086 | 3.6e-4 | `2⁻¹⁴` |
| `p = 1e-2`, 20 steps, 5π/16 | +0.002483689686 | 5.1e-5 | `2⁻¹⁶` |
| `p = 3e-2`, 20 steps, 5π/16 | +0.000077952653 | 4.2e-7 | `2⁻²⁰` |

**Not claimable, and why.** Two kinds, and the driver prints both:

* `p = 1e-3`, 20 steps — the cutoff sweep never plateaued (§2.1). The value it reports, 0.012998, is
  probably close; "probably close" is not a claim, and the grid cannot be tightened further at that
  `p` inside this showcase's time box (its `2⁻¹⁴` point already costs 361 s).
* `θ_h = 7π/32`, 20 steps, noiseless — **Benchmark C's measured reachability boundary.** C's
  reference sweep reached 3.9e7 terms at `2⁻¹⁶` and the value still moved by **1.44e-1** on the last
  tightening; C projects `2⁻²⁰`–`2⁻²²`, ~1e10 terms, ~560 GiB and ~17 h at 32 threads to reach the
  plan's 0.01 bar, and notes that the published record contains no exact 20-step value either. C
  reports it as not claimable, and so does this showcase — see
  [`../../benchmarks/python/deep_trotter/README.md`](../../benchmarks/python/deep_trotter/README.md)
  §3.1–3.3. Nothing in B2 improves the *noiseless* answer there.

### 5.1 The reachability boundary, with and without noise

The interesting question is what noise does to that boundary, so Part 4 runs the same 7π/32,
20-step circuit at `p = 1e-2`:

| cutoff | ⟨Z₆₂⟩ | final terms | peak terms | wall | Δ vs previous |
|---|---|---|---|---|---|
| `2⁻¹⁰` | +0.111648251336 | 156 | 3 550 | 0.25 s | — |
| `2⁻¹²` | +0.111517918895 | 1 313 | 19 569 | 1.16 s | 1.30e-4 |
| `2⁻¹⁴` | +0.110439488211 | 10 918 | 130 826 | 6.77 s | 1.08e-3 |
| `2⁻¹⁶` | +0.109507564808 | 74 717 | 937 075 | 50.2 s | **9.32e-4** |

At the **same** `2⁻¹⁶` cutoff, noiseless (C) against noisy (here):

| | peak terms | still resident at the end | last Δ |
|---|---|---|---|
| noiseless (Benchmark C) | 45.4 million | 38.8 million | 1.44e-1 |
| `p = 1e-2` (this run) | 0.94 million | 74 717 | 9.32e-4 |

**48× fewer peak terms (519× fewer still resident at the end), and a 154× smaller last
difference.** And it still does not pass: the plateau test
needs the last *two* differences under `1e-3`, and the second-to-last is 1.08e-3. So the verdict
stays `converged = false` and the value stays unclaimable — the criterion is not bent to fit the
result. One more dyadic pair (`2⁻¹⁸`, projected ~6 min on the measured ~7× per-pair growth) would
very likely settle it; it was cut, since the comparison above is the point and does not need it.

Two caveats, both load-bearing:

* **This is a different quantity.** `⟨Z₆₂⟩` under the noisy channel is not `⟨Z₆₂⟩` under the unitary
  circuit; they agree only as `p → 0` and at `p = 1e-2` after 20 steps they are far apart (0.1095
  against C's unresolved ~0.25–0.52 band). Noise is not a cheaper route to the noiseless number.
* **But it is the physically relevant one for a verification claim.** A device running this circuit
  has noise; the quantity a classical simulation should be asked to reproduce is the noisy one. That
  regime is precisely the cheap one here — which is the practical form of §2's headline.

## 6. Validation: an independent dense noisy reference

Every oracle in `examples/common/oracles.py` **refuses** a noise channel by construction — the
statevector oracle because it is unitary-only, the stim oracle because a tableau simulation samples
one Pauli error rather than averaging over them — and both say so pointing at "a density-matrix
reference, which this module does not provide". So B2's correctness gate provides one, hand-rolled
in the test file:

* the circuit's gates are applied to an explicit `2^n × 2^n` density matrix, forward in time from
  `|0…0⟩⟨0…0|`, unitaries by conjugation and channels as `Σ_k K_k ρ K_k†` with the Kraus operators
  written from each channel's *definition* (not from the engine's dual);
* two-qubit Cliffords are expressed as sums of embedded Pauli products
  (`CNOT = (I + Z_c + X_t − Z_c X_t)/2`, `SWAP = (II + XX + YY + ZZ)/2`), so the reference never has
  to reason about tensor-factor ordering for a non-adjacent pair — the one place a dense cross-check
  is easy to get silently wrong;
* the reference is itself guarded: with no noise gate it must reproduce the engine's exact
  untruncated Pauli propagation.

What it checks, at `1e-10`, for all five channels (`depolarize`, `dephase`, `amplitude_damping`,
`pauli_channel`, `depolarize2`) mixed with `h`/`s`/`cnot`/`cz`/`swap` and non-Clifford `rx`/`ry`/`rz`
plus a weight-two `XY` rotation:

| direction | what is compared | size |
|---|---|---|
| `heisenberg` | `⟨0…0\|Φ†(O)\|0…0⟩` from the engine against `Tr(O Φ(ρ₀))` from the Kraus evolution | `n = 6` |
| `forward` | **every one of the `4^n` Pauli coefficients** of the engine's evolved `ρ` against the dense `ρ`'s, so a *missing* term fails too | `n = 4` |

Both directions matter because `amplitude_damping` is the one channel here that is neither
self-adjoint nor key-preserving: its Schrödinger map is `I → I + γZ`, `Z → (1−γ)Z`, its Heisenberg
dual is `I → I`, `Z → (1−γ)Z + γI` — it *fans out* while it damps — and the two were swapped in the
core until commit `e42095c`. `test_amplitude_damping_drives_z_to_the_plus_one_fixed_point` pins the
physics that fix turned on: `⟨Z⟩` for a qubit already in `|0⟩` (the channel's fixed point) stays
exactly 1 for every `γ`, and from `|1⟩` it lands on `2γ − 1`.

The gate also pins the mechanism (§1.1) and the citation (§4): 29 tests, ~1 s, 97 MiB peak RSS,
numpy-only, no `importorskip`.

## 7. Recorded cuts

Plan §6/D15's time-box policy is "pilot, project, then shrink the grid and record the cut". The cuts
are the table `COEFF_GRID_CUTS` in the driver, not an adaptive stop, so they are reviewable and do
not change shape with machine load. Their reasons are in that table verbatim; the pilot they come
from is `MEASURED_PILOT` (ccqlin038, single-threaded, `θ_h = 5π/16`, 20 steps, 13 720 channels):

| p | cutoff | ⟨Z₆₂⟩ | final terms | peak terms | wall |
|---|---|---|---|---|---|
| 0 | 2⁻¹⁰ | +0.014729671765 | 119 | 79 029 | 2.45 s |
| 0 | 2⁻¹² | +0.015481385131 | 2 543 | 1 071 093 | 37.9 s |
| 1e-3 | 2⁻¹⁴ | +0.012998050638 | 20 098 | 9 710 246 | 380 s |
| 5e-3 | 2⁻¹⁴ | +0.006280876086 | 3 521 | 2 818 675 | 83.5 s |
| 1e-2 | 2⁻¹⁴ | +0.002432260559 | 408 | 869 299 | 20.2 s |
| 1e-2 | 2⁻¹⁶ | +0.002483689686 | 5 063 | 6 445 313 | 167 s |
| 3e-2 | 2⁻¹⁴ | +0.000000000000 | 0 | 22 105 | 0.52 s |
| 3e-2 | 2⁻¹⁸ | +0.000077529951 | 70 | 738 435 | 12.4 s |

Three readings decide the cuts:

* **The `p = 0` leg is the whole cost of the sweep.** `2⁻¹²` alone is 38 s and `2⁻¹⁴` is minutes.
  Benchmark C measured the same point and stopped there too, its plateau test satisfied. The next
  pair projects, from this leg's own measured growth, to ~1.9e8 terms (~9 GiB of columns) and ~2 h
  single-threaded — past this showcase's entire box for one grid point. Cut at `2⁻¹⁴`.
* **The cheapness is monotone in `p`, so the *reach* is too.** `2⁻¹⁴` costs 380 s at `p = 1e-3`,
  84 s at `5e-3`, 20 s at `1e-2` and half a second at `3e-2`. The cuts follow: `2⁻¹⁴` for
  `p ≤ 5e-3`, `2⁻¹⁶` for `1e-2`, `2⁻²⁰` for `3e-2`.
* **A third kick angle was cut.** `θ_h = 7π/32` gets one noisy leg (§5) rather than a full `p` grid:
  its `p = 0` column would cost another ~20 min and Benchmark C already established that the
  noiseless answer there is unreachable, which is the only thing the second angle is used for here.

The `p = 3e-2` leg is deliberately *not* cut at the marquee cutoff even though its tracked set
empties there — see §2.1.

## 8. Caveats

* **Wall times are indicative, not campaign-grade.** Single-threaded on a shared workstation whose
  stated single-thread campaign noise is ±5–8 % (CLAUDE.md §Performance discipline), and taken with
  `warmup=False`: the tight-cutoff `p = 0` leg costs minutes, so a discarded warm-up pass would
  double the whole run. Term counts, expectation values and convergence outcomes are
  load-independent and are the numbers to quote. Anything under ~10 % needs
  `scripts/ab-compare.sh`, not this driver.
* **The 20-step noiseless row is a reproducibility check, not an accuracy one.** Benchmark C's
  20-step reference at `θ_h = 5π/16` is *self-converged* — C's own tightest run — so reproducing it
  says the noise plumbing at `p = 0` does not perturb the answer, not that the answer is right. The
  accuracy statements in §4 rest on the two 5-step rows, whose references are exact (a causal-cone
  reduction cross-checked between an Aer statevector and an untruncated Pauli propagation).
* **A truncated Pauli sum has no variational bound.** Discarded terms carry signs, so a partial sum
  can sit on either side of the truth and the error need not be monotone in the cutoff — C measured
  exactly that at `θ_h = 7π/32`. That is why every `p` here carries a cutoff sweep and a plateau
  verdict rather than a single number, and why `converged = false` must be read as "no usable
  uncertainty estimate", not "a slightly weaker one".
* **The noise model is a model, not a device.** One depolarizing channel per gate-qubit with a
  single rate is the standard first-order stand-in; a real Eagle device has different one- and
  two-qubit error rates, coherent errors, crosstalk, and readout error, none of which is here.
  `p = 1e-3` is the order of a good two-qubit gate error; `1e-2` and `3e-2` are deliberately past it,
  chosen to show the collapse rather than to model hardware.
* **`min_abs_coeff ≥ 1e-12` everywhere**, inherited from Benchmark B's `MIN_SAFE_COEFF`:
  `cos(π/2) == 6.123233995736766e-17`, not zero, so at the Clifford `θ_zz` every rotation leaves a
  numerically-dead residual branch and an untruncated 127-qubit propagation fans out without bound.
  `run_one` refuses a smaller cutoff.
* **Provenance: the recorded run's `dirty` flag is `true`.** At run time the worktree carried this
  (then-untracked) showcase directory plus a two-line edit to `examples/README.md`, and two
  comment-only edits landed in `run_b2.py` while the run was in flight. No executable line differs
  from the committed driver, and every value, term count and convergence verdict here is identical
  to an earlier full run of the same driver — only wall times moved, by ≤ 2 %.
* **Peak, not final, is the cost.** At the noisier settings the two differ by orders of magnitude
  (the sum blows up mid-circuit and collapses by the end), and it is the peak a run has to hold.
  `results.json` carries both.
